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

#include "itkGradientDescentOptimizer2.h"

#include "itkCommand.h"
#include "itkEventObject.h"
#include "itkMacro.h"

#ifdef ELASTIX_USE_OPENMP
#  include <omp.h>
#endif


namespace itk
{

/**
 * ****************** Constructor ************************
 */

GradientDescentOptimizer2 ::GradientDescentOptimizer2()
{
  itkDebugMacro("Constructor");

  this->m_Stop = false;
  this->m_LearningRate = 1.0;
  this->m_NumberOfIterations = 100;
  this->m_CurrentIteration = 0;
  this->m_Value = 0.0;
  this->m_StopCondition = MaximumNumberOfIterations;

  this->m_UseOpenMP = false;
#ifdef ELASTIX_USE_OPENMP
  this->m_UseOpenMP = true;
#endif

} // end Constructor


/**
 * *************** PrintSelf *************************
 */

void
GradientDescentOptimizer2 ::PrintSelf(std::ostream & os, Indent indent) const
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
GradientDescentOptimizer2 ::StartOptimization(void)
{
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
GradientDescentOptimizer2 ::ResumeOptimization(void)
{
  itkDebugMacro("ResumeOptimization");

  this->m_Stop = false;

  InvokeEvent(StartEvent());

  const unsigned int spaceDimension = this->GetScaledCostFunction()->GetNumberOfParameters();
  this->m_Gradient = DerivativeType(spaceDimension); // check this

  while (!this->m_Stop)
  {
    try
    {
      this->GetScaledValueAndDerivative(this->GetScaledCurrentPosition(), m_Value, m_Gradient);
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
GradientDescentOptimizer2 ::MetricErrorResponse(ExceptionObject & err)
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
GradientDescentOptimizer2 ::StopOptimization(void)
{
  itkDebugMacro("StopOptimization");

  this->m_Stop = true;
  this->InvokeEvent(EndEvent());
} // end StopOptimization()


/**
 * ************ AdvanceOneStep ****************************
 */

void
GradientDescentOptimizer2 ::AdvanceOneStep(void)
{
  itkDebugMacro("AdvanceOneStep");

  /** Get space dimension. */
  const unsigned int spaceDimension = this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Get a reference to the previously allocated newPosition. */
  ParametersType & newPosition = this->m_ScaledCurrentPosition;

  /** Advance one step. */
#if 1 // force single-threaded since it is fastest most of the times
      //#ifndef ELASTIX_USE_OPENMP // If no OpenMP detected then use single-threaded code
  /** Get a reference to the current position. */
  const ParametersType & currentPosition = this->GetScaledCurrentPosition();

  /** Update the new position. */
  for (unsigned int j = 0; j < spaceDimension; ++j)
  {
    newPosition[j] = currentPosition[j] - this->m_LearningRate * this->m_Gradient[j];
  }
#else // Otherwise use OpenMP
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
#endif

  this->InvokeEvent(IterationEvent());

} // end AdvanceOneStep()


} // end namespace itk
