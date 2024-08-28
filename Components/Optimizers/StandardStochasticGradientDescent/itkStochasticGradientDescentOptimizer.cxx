/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "itkStochasticGradientDescentOptimizer.h"
#include "itkCommand.h"
#include "itkEventObject.h"
#include "itkMacro.h"

#ifdef ELASTIX_USE_EIGEN
#  include <Eigen/Dense>
#  include <Eigen/Core>
#endif

#include <algorithm> // For min.
#include <cassert>

namespace itk
{

/**
 * ****************** Constructor ************************
 */

StochasticGradientDescentOptimizer::StochasticGradientDescentOptimizer()
{
  itkDebugMacro("Constructor");

} // end Constructor


/**
 * *************** PrintSelf *************************
 */

void
StochasticGradientDescentOptimizer::PrintSelf(std::ostream & os, Indent indent) const
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "LearningRate: " << this->m_LearningRate << std::endl;
  os << indent << "NumberOfIterations: " << this->m_NumberOfIterations << std::endl;
  os << indent << "CurrentIteration: " << this->m_CurrentIteration;
  os << indent << "Value: " << this->m_Value;
  os << indent << "StopCondition: " << this->m_StopCondition;
  os << std::endl;
  os << indent << "Gradient: " << this->m_Gradient;
  os << std::endl;

} // end PrintSelf()


/**
 * **************** StartOptimization ********************
 */

void
StochasticGradientDescentOptimizer::StartOptimization()
{
  itkDebugMacro("StartOptimization");

  this->m_CurrentIteration = 0;

  /** Get the number of parameters; checks also if a cost function has been set at all.
   * if not: an exception is thrown */
  this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Initialize the scaledCostFunction with the currently set scales */
  this->InitializeScales();

  /** Set the current position as the scaled initial position */
  this->SetCurrentPosition(this->GetInitialPosition());

  this->ResumeOptimization();
} // end StartOptimization()


/**
 * ************************ ResumeOptimization *************
 */

void
StochasticGradientDescentOptimizer::ResumeOptimization()
{
  itkDebugMacro("ResumeOptimization");

  this->m_Stop = false;

  InvokeEvent(StartEvent());

  this->m_PreviousGradient = this->GetPreviousGradient();
  this->m_PreviousPosition = this->GetPreviousPosition();

  const unsigned int spaceDimension = this->GetScaledCostFunction()->GetNumberOfParameters();
  this->m_Gradient.set_size(spaceDimension); // check this

  DerivativeType currentPositionGradient;
  DerivativeType previousPositionGradient;

  while (!this->m_Stop)
  {
    if (m_CurrentIteration >= m_NumberOfIterations)
    {
      // Check m_CurrentIteration right at the start of the loop, ensuring that
      // no step at all is performed when when m_NumberOfIterations is zero.
      this->m_StopCondition = MaximumNumberOfIterations;
      this->StopOptimization();
      break;
    }

    try
    {
      this->GetScaledValueAndDerivative(this->GetScaledCurrentPosition(), m_Value, this->m_Gradient);
    }
    catch (ExceptionObject & err)
    {
      this->MetricErrorResponse(err);
    }

    /** StopOptimization may have been called. */
    if (this->m_Stop)
    {
      break;
    }

    this->AdvanceOneStep();

    /** StopOptimization may have been called. */
    if (this->m_Stop)
    {
      break;
    }

    this->m_CurrentIteration++;

  } // end while

} // end ResumeOptimization()


/**
 * ***************** MetricErrorResponse ************************
 */

void
StochasticGradientDescentOptimizer::MetricErrorResponse(ExceptionObject & err)
{
  /** An exception has occurred. Terminate immediately. */
  this->m_StopCondition = MetricError;
  this->StopOptimization();

  /** Pass exception to caller. */
  throw err;

} // end MetricErrorResponse()


/**
 * ***************** Stop optimization ************************
 */

void
StochasticGradientDescentOptimizer::StopOptimization()
{
  itkDebugMacro("StopOptimization");

  this->m_Stop = true;
  this->InvokeEvent(EndEvent());
} // end StopOptimization


/**
 * ************ AdvanceOneStep ****************************
 */

void
StochasticGradientDescentOptimizer::AdvanceOneStep()
{
  itkDebugMacro("AdvanceOneStep");

  /** Get space dimension. */
  const unsigned int spaceDimension = this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Get a reference to the previously allocated newPosition. */
  ParametersType & newPosition = this->m_ScaledCurrentPosition;

  /** Advance one step. */
  // for now force single-threaded since it is fastest most of the times
  /** Get a reference to the current position. */
  const ParametersType & currentPosition = this->GetScaledCurrentPosition();

  /** Update the new position. */
  for (unsigned int j = 0; j < spaceDimension; ++j)
  {
    newPosition[j] = currentPosition[j] - this->m_LearningRate * this->m_Gradient[j];
  }

  this->InvokeEvent(IterationEvent());

} // end AdvanceOneStep()


/**
 * ************ AdvanceOneStepThreaderCallback ****************************
 */

ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
StochasticGradientDescentOptimizer::AdvanceOneStepThreaderCallback(void * arg)
{
  /** Get the current thread id and user data. */
  assert(arg);
  const auto & infoStruct = *static_cast<ThreadInfoType *>(arg);
  ThreadIdType threadID = infoStruct.WorkUnitID;

  assert(infoStruct.UserData);
  const auto & userData = *static_cast<MultiThreaderParameterType *>(infoStruct.UserData);

  /** Call the real implementation. */
  userData.t_Optimizer->ThreadedAdvanceOneStep(threadID, *(userData.t_NewPosition));

  return ITK_THREAD_RETURN_DEFAULT_VALUE;

} // end AdvanceOneStepThreaderCallback()


/**
 * ************ ThreadedAdvanceOneStep ****************************
 */

void
StochasticGradientDescentOptimizer::ThreadedAdvanceOneStep(ThreadIdType threadId, ParametersType & newPosition)
{
  /** Compute the range for this thread. */
  const unsigned int spaceDimension = this->GetScaledCostFunction()->GetNumberOfParameters();
  const auto         subSize = static_cast<unsigned int>(
    std::ceil(static_cast<double>(spaceDimension) / static_cast<double>(this->m_Threader->GetNumberOfWorkUnits())));
  const unsigned int jmin = threadId * subSize;
  const unsigned int jmax = std::min((threadId + 1) * subSize, spaceDimension);

  /** Get a reference to the current position. */
  const ParametersType & currentPosition = this->GetScaledCurrentPosition();
  const double           learningRate = this->m_LearningRate;
  const DerivativeType & gradient = this->m_Gradient;

  /** Advance one step: mu_{k+1} = mu_k - a_k * gradient_k */
  for (unsigned int j = jmin; j < jmax; ++j)
  {
    newPosition[j] = currentPosition[j] - learningRate * gradient[j];
  }

} // end ThreadedAdvanceOneStep()


} // end namespace itk
