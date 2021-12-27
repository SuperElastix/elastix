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
#ifndef elxStandardGradientDescent_hxx
#define elxStandardGradientDescent_hxx

#include "elxStandardGradientDescent.h"
#include <iomanip>
#include <string>

namespace elastix
{

/**
 * ***************** Constructor ***********************
 */

template <class TElastix>
StandardGradientDescent<TElastix>::StandardGradientDescent()
{
  this->m_MaximumNumberOfSamplingAttempts = 0;
  this->m_CurrentNumberOfSamplingAttempts = 0;
  this->m_PreviousErrorAtIteration = 0;

} // end Constructor()


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void
StandardGradientDescent<TElastix>::BeforeRegistration()
{
  /** Add the target cell "stepsize" to IterationInfo.*/
  this->AddTargetCellToIterationInfo("2:Metric");
  this->AddTargetCellToIterationInfo("3:StepSize");
  this->AddTargetCellToIterationInfo("4:||Gradient||");

  /** Format the metric and stepsize as floats */
  this->GetIterationInfoAt("2:Metric") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("3:StepSize") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("4:||Gradient||") << std::showpoint << std::fixed;

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
StandardGradientDescent<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level. */
  unsigned int level = static_cast<unsigned int>(this->m_Registration->GetAsITKBaseType()->GetCurrentLevel());

  /** Set the maximumNumberOfIterations. */
  unsigned int maximumNumberOfIterations = 500;
  this->GetConfiguration()->ReadParameter(
    maximumNumberOfIterations, "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0);
  this->SetNumberOfIterations(maximumNumberOfIterations);

  /** Set the gain parameters */
  double a = 400.0;
  double A = 50.0;
  double alpha = 0.602;

  this->GetConfiguration()->ReadParameter(a, "SP_a", this->GetComponentLabel(), level, 0);
  this->GetConfiguration()->ReadParameter(A, "SP_A", this->GetComponentLabel(), level, 0);
  this->GetConfiguration()->ReadParameter(alpha, "SP_alpha", this->GetComponentLabel(), level, 0);

  this->SetParam_a(a);
  this->SetParam_A(A);
  this->SetParam_alpha(alpha);

  /** Set the MaximumNumberOfSamplingAttempts. */
  unsigned int maximumNumberOfSamplingAttempts = 0;
  this->GetConfiguration()->ReadParameter(
    maximumNumberOfSamplingAttempts, "MaximumNumberOfSamplingAttempts", this->GetComponentLabel(), level, 0);
  this->SetMaximumNumberOfSamplingAttempts(maximumNumberOfSamplingAttempts);
  if (maximumNumberOfSamplingAttempts > 5)
  {
    elxout["warning"] << "\nWARNING: You have set MaximumNumberOfSamplingAttempts to "
                      << maximumNumberOfSamplingAttempts << ".\n"
                      << "  This functionality is known to cause problems (stack overflow) for large values.\n"
                      << "  If elastix stops or segfaults for no obvious reason, reduce this value.\n"
                      << "  You may select the RandomSparseMask image sampler to fix mask-related problems.\n"
                      << std::endl;
  }

} // end BeforeEachResolution()


/**
 * ***************** AfterEachIteration *************************
 */

template <class TElastix>
void
StandardGradientDescent<TElastix>::AfterEachIteration()
{
  /** Print some information */
  this->GetIterationInfoAt("2:Metric") << this->GetValue();
  this->GetIterationInfoAt("3:StepSize") << this->GetLearningRate();
  this->GetIterationInfoAt("4:||Gradient||") << this->GetGradient().magnitude();

  /** Select new spatial samples for the computation of the metric */
  if (this->GetNewSamplesEveryIteration())
  {
    this->SelectNewSamples();
  }

} // end AfterEachIteration()


/**
 * ***************** AfterEachResolution *************************
 */

template <class TElastix>
void
StandardGradientDescent<TElastix>::AfterEachResolution()
{
  /**
   * enum   StopConditionType {  MaximumNumberOfIterations, MetricError }
   */
  std::string stopcondition;
  switch (this->GetStopCondition())
  {

    case MaximumNumberOfIterations:
      stopcondition = "Maximum number of iterations has been reached";
      break;

    case MetricError:
      stopcondition = "Error in metric";
      break;

    default:
      stopcondition = "Unknown";
      break;
  }

  /** Print the stopping condition */
  elxout << "Stopping condition: " << stopcondition << "." << std::endl;

} // end AfterEachResolution()


/**
 * ******************* AfterRegistration ************************
 */

template <class TElastix>
void
StandardGradientDescent<TElastix>::AfterRegistration()
{
  /** Print the best metric value */
  double bestValue = this->GetValue();
  elxout << '\n' << "Final metric value  = " << bestValue << std::endl;

} // end AfterRegistration()


/**
 * ****************** StartOptimization *************************
 */

template <class TElastix>
void
StandardGradientDescent<TElastix>::StartOptimization()
{
  /** Check if the entered scales are correct and != [ 1 1 1 ...] */
  this->SetUseScales(false);
  const ScalesType & scales = this->GetScales();
  if (scales.GetSize() == this->GetInitialPosition().GetSize())
  {
    ScalesType unit_scales(scales.GetSize());
    unit_scales.Fill(1.0);
    if (scales != unit_scales)
    {
      /** only then: */
      this->SetUseScales(true);
    }
  }

  /** Reset these values. */
  this->m_CurrentNumberOfSamplingAttempts = 0;
  this->m_PreviousErrorAtIteration = 0;

  /** Superclass implementation. */
  this->Superclass1::StartOptimization();

} // end StartOptimization()


/**
 * ****************** MetricErrorResponse *************************
 */

template <class TElastix>
void
StandardGradientDescent<TElastix>::MetricErrorResponse(itk::ExceptionObject & err)
{
  if (this->GetCurrentIteration() != this->m_PreviousErrorAtIteration)
  {
    this->m_PreviousErrorAtIteration = this->GetCurrentIteration();
    this->m_CurrentNumberOfSamplingAttempts = 1;
  }
  else
  {
    this->m_CurrentNumberOfSamplingAttempts++;
  }

  if (this->m_CurrentNumberOfSamplingAttempts <= this->m_MaximumNumberOfSamplingAttempts)
  {
    this->SelectNewSamples();
    this->ResumeOptimization();
  }
  else
  {
    /** Stop optimisation and pass on exception. */
    this->Superclass1::MetricErrorResponse(err);
  }

} // end MetricErrorResponse()


} // end namespace elastix

#endif // end #ifndef elxStandardGradientDescent_hxx
