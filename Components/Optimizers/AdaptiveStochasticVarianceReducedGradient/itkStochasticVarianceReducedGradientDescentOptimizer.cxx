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

#include "itkStochasticVarianceReducedGradientDescentOptimizer.h"

#include "itkCommand.h"
#include "itkEventObject.h"
#include "itkMacro.h"

#ifdef ELASTIX_USE_OPENMP
#  include <omp.h>
#endif

#ifdef ELASTIX_USE_EIGEN
#  include <Eigen/Dense>
#  include <Eigen/Core>
#endif

namespace itk
{

/**
 * ****************** Constructor ************************
 */

StochasticVarianceReducedGradientDescentOptimizer::StochasticVarianceReducedGradientDescentOptimizer()
{
  itkDebugMacro("Constructor");

  this->m_LearningRate = 1.0;
  this->m_NumberOfIterations = 100;
  this->m_CurrentIteration = 0;
  this->m_LBFGSMemory = 0;
  this->m_Value = 0.0;
  this->m_StopCondition = MaximumNumberOfIterations;

  this->m_Threader = ThreaderType::New();
  this->m_UseMultiThread = false;
  this->m_UseOpenMP = false;
  this->m_UseEigen = false;

} // end Constructor


/**
 * *************** PrintSelf *************************
 */

void
StochasticVarianceReducedGradientDescentOptimizer::PrintSelf(std::ostream & os, Indent indent) const
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
StochasticVarianceReducedGradientDescentOptimizer::StartOptimization(void)
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
StochasticVarianceReducedGradientDescentOptimizer::ResumeOptimization(void)
{
  itkDebugMacro("ResumeOptimization");

  this->m_Stop = false;

  InvokeEvent(StartEvent());

  this->m_PreviousGradient = this->GetPreviousGradient();
  this->m_PreviousPosition = this->GetPreviousPosition();

  const unsigned int spaceDimension = this->GetScaledCostFunction()->GetNumberOfParameters();
  this->m_Gradient = DerivativeType(spaceDimension); // check this

  DerivativeType currentPositionGradient;
  DerivativeType previousPositionGradient;

  while (!this->m_Stop)
  {
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

    if (m_CurrentIteration >= m_NumberOfIterations)
    {
      this->m_StopCondition = MaximumNumberOfIterations;
      this->StopOptimization();
      break;
    }

  } // end while

} // end ResumeOptimization()


/**
 * ***************** MetricErrorResponse ************************
 */

void
StochasticVarianceReducedGradientDescentOptimizer::MetricErrorResponse(ExceptionObject & err)
{
  /** An exception has occurred. Terminate immediately. */
  this->m_StopCondition = MetricError;
  this->StopOptimization();

  /** Pass exception to caller. */
  throw err;

} // end MetricErrorResponse()


/**
 * ***************** StopOptimization ************************
 */

void
StochasticVarianceReducedGradientDescentOptimizer::StopOptimization(void)
{
  itkDebugMacro("StopOptimization");

  this->m_Stop = true;
  this->InvokeEvent(EndEvent());
} // end StopOptimization()


/**
 * ************ AdvanceOneStep ****************************
 */

void
StochasticVarianceReducedGradientDescentOptimizer::AdvanceOneStep(void)
{
  itkDebugMacro("AdvanceOneStep");

  /** Get space dimension. */
  const unsigned int spaceDimension = this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Get a reference to the previously allocated newPosition. */
  ParametersType & newPosition = this->m_ScaledCurrentPosition;

  /** Advance one step. */
  // single-threadedly
  if (!this->m_UseMultiThread || true) // for now force single-threaded since it is fastest most of the times
  // if( !this->m_UseMultiThread && false ) // force multi-threaded
  {
    /** Get a reference to the current position. */
    const ParametersType & currentPosition = this->GetScaledCurrentPosition();

    /** Update the new position. */
    for (unsigned int j = 0; j < spaceDimension; j++)
    {
      newPosition[j] = currentPosition[j] - this->m_LearningRate * this->m_Gradient[j];
    }
  }
#ifdef ELASTIX_USE_OPENMP
  else if (this->m_UseOpenMP && !this->m_UseEigen)
  {
    /** Get a reference to the current position. */
    const ParametersType & currentPosition = this->GetScaledCurrentPosition();

    /** Update the new position. */
    const int nthreads = static_cast<int>(this->m_Threader->GetNumberOfWorkUnits());
    omp_set_num_threads(nthreads);
#  pragma omp parallel for
    for (int j = 0; j < static_cast<int>(spaceDimension); j++)
    {
      newPosition[j] = currentPosition[j] - this->m_LearningRate * this->m_Gradient[j];
    }
  }
#endif
#ifdef ELASTIX_USE_EIGEN
  else if (!this->m_UseOpenMP && this->m_UseEigen)
  {
    /** Get a reference to the current position. */
    const ParametersType & currentPosition = this->GetScaledCurrentPosition();
    const double           learningRate = this->m_LearningRate;

    /** Wrap itk::Arrays into Eigen jackets. */
    typedef Eigen::VectorXd               ParametersTypeEigen;
    Eigen::Map<ParametersTypeEigen>       newPositionE(newPosition.data_block(), spaceDimension);
    Eigen::Map<const ParametersTypeEigen> currentPositionE(currentPosition.data_block(), spaceDimension);
    Eigen::Map<ParametersTypeEigen>       gradientE(this->m_Gradient.data_block(), spaceDimension);

    /** Update the new position. */
    newPositionE = currentPositionE - learningRate * gradientE;
  }
#endif
#if defined(ELASTIX_USE_OPENMP) && defined(ELASTIX_USE_EIGEN)
  else if (this->m_UseOpenMP && this->m_UseEigen)
  {
    /** Get a reference to the current position. */
    const ParametersType & currentPosition = this->GetScaledCurrentPosition();
    const double           learningRate = this->m_LearningRate;

    /** Wrap itk::Arrays into Eigen jackets. */
    typedef Eigen::VectorXd               ParametersTypeEigen;
    Eigen::Map<ParametersTypeEigen>       newPositionE(newPosition.data_block(), spaceDimension);
    Eigen::Map<const ParametersTypeEigen> currentPositionE(currentPosition.data_block(), spaceDimension);
    Eigen::Map<ParametersTypeEigen>       gradientE(this->m_Gradient.data_block(), spaceDimension);

    /** Update the new position. */
    const int spaceDim = static_cast<int>(spaceDimension);
    const int nthreads = static_cast<int>(this->m_Threader->GetNumberOfWorkUnits());
    omp_set_num_threads(nthreads);
#  pragma omp parallel for
    for (int i = 0; i < nthreads; i += 1)
    {
      int threadId = omp_get_thread_num();
      int chunk = (spaceDimension + nthreads - 1) / nthreads;
      int jmin = threadId * chunk;
      int jmax = (threadId + 1) * chunk < spaceDim ? (threadId + 1) * chunk : spaceDim;
      int subSize = jmax - jmin;

      newPositionE.segment(jmin, subSize) =
        currentPositionE.segment(jmin, subSize) - learningRate * gradientE.segment(jmin, subSize);
    }
  }
#endif
  else
  {
    /** Fill the threader parameter struct with information. */
    MultiThreaderParameterType * temp = new MultiThreaderParameterType;
    temp->t_NewPosition = &newPosition;
    temp->t_Optimizer = this;

    /** Call multi-threaded AdvanceOneStep(). */
    ThreaderType::Pointer local_threader = ThreaderType::New();
    local_threader->SetNumberOfWorkUnits(this->m_Threader->GetNumberOfWorkUnits());
    local_threader->SetSingleMethod(AdvanceOneStepThreaderCallback, temp);
    local_threader->SingleMethodExecute();

    delete temp;
  }

  this->InvokeEvent(IterationEvent());

} // end AdvanceOneStep()


/**
 * ************ AdvanceOneStepThreaderCallback ****************************
 */

ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
StochasticVarianceReducedGradientDescentOptimizer::AdvanceOneStepThreaderCallback(void * arg)
{
  /** Get the current thread id and user data. */
  ThreadInfoType *             infoStruct = static_cast<ThreadInfoType *>(arg);
  ThreadIdType                 threadID = infoStruct->WorkUnitID;
  MultiThreaderParameterType * temp = static_cast<MultiThreaderParameterType *>(infoStruct->UserData);

  /** Call the real implementation. */
  temp->t_Optimizer->ThreadedAdvanceOneStep(threadID, *(temp->t_NewPosition));

  return itk::ITK_THREAD_RETURN_DEFAULT_VALUE;

} // end AdvanceOneStepThreaderCallback()


/**
 * ************ ThreadedAdvanceOneStep ****************************
 */

void
StochasticVarianceReducedGradientDescentOptimizer::ThreadedAdvanceOneStep(ThreadIdType     threadId,
                                                                          ParametersType & newPosition)
{
  /** Compute the range for this thread. */
  const unsigned int spaceDimension = this->GetScaledCostFunction()->GetNumberOfParameters();
  const unsigned int subSize = static_cast<unsigned int>(
    std::ceil(static_cast<double>(spaceDimension) / static_cast<double>(this->m_Threader->GetNumberOfWorkUnits())));
  const unsigned int jmin = threadId * subSize;
  unsigned int       jmax = (threadId + 1) * subSize;
  jmax = (jmax > spaceDimension) ? spaceDimension : jmax;

  /** Get a reference to the current position. */
  const ParametersType & currentPosition = this->GetScaledCurrentPosition();
  const double           learningRate = this->m_LearningRate;
  const DerivativeType & gradient = this->m_Gradient;

  /** Advance one step: mu_{k+1} = mu_k - a_k * gradient_k */
  for (unsigned int j = jmin; j < jmax; j++)
  {
    newPosition[j] = currentPosition[j] - learningRate * gradient[j];
  }

} // end ThreadedAdvanceOneStep()


} // end namespace itk
