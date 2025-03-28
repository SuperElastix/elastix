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
#include <itkDeref.h>
#include <iomanip>
#include <string>

namespace elastix
{

/**
 * ***************** Constructor ***********************
 */

template <typename TElastix>
StandardGradientDescent<TElastix>::StandardGradientDescent()
{
  this->m_MaximumNumberOfSamplingAttempts = 0;
  this->m_CurrentNumberOfSamplingAttempts = 0;
  this->m_PreviousErrorAtIteration = 0;

} // end Constructor()


/**
 * ***************** BeforeRegistration ***********************
 */

template <typename TElastix>
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

template <typename TElastix>
void
StandardGradientDescent<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level. */
  auto level = static_cast<unsigned int>(this->m_Registration->GetAsITKBaseType()->GetCurrentLevel());

  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

  /** Set the maximumNumberOfIterations. */
  unsigned int maximumNumberOfIterations = 500;
  configuration.ReadParameter(
    maximumNumberOfIterations, "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0);
  this->SetNumberOfIterations(maximumNumberOfIterations);

  /** Set the gain parameters */
  double a = 400.0;
  double A = 50.0;
  double alpha = 0.602;

  configuration.ReadParameter(a, "SP_a", this->GetComponentLabel(), level, 0);
  configuration.ReadParameter(A, "SP_A", this->GetComponentLabel(), level, 0);
  configuration.ReadParameter(alpha, "SP_alpha", this->GetComponentLabel(), level, 0);

  this->SetParam_a(a);
  this->SetParam_A(A);
  this->SetParam_alpha(alpha);

  /** Set the MaximumNumberOfSamplingAttempts. */
  unsigned int maximumNumberOfSamplingAttempts = 0;
  configuration.ReadParameter(
    maximumNumberOfSamplingAttempts, "MaximumNumberOfSamplingAttempts", this->GetComponentLabel(), level, 0);
  this->SetMaximumNumberOfSamplingAttempts(maximumNumberOfSamplingAttempts);
  if (maximumNumberOfSamplingAttempts > 5)
  {
    log::warn(
      std::ostringstream{} << "\nWARNING: You have set MaximumNumberOfSamplingAttempts to "
                           << maximumNumberOfSamplingAttempts << ".\n"
                           << "  This functionality is known to cause problems (stack overflow) for large values.\n"
                           << "  If elastix stops or segfaults for no obvious reason, reduce this value.\n"
                           << "  You may select the RandomSparseMask image sampler to fix mask-related problems.\n");
  }

} // end BeforeEachResolution()


/**
 * ***************** AfterEachIteration *************************
 */

template <typename TElastix>
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

template <typename TElastix>
void
StandardGradientDescent<TElastix>::AfterEachResolution()
{
  /**
   * enum   StopConditionType {  MaximumNumberOfIterations, MetricError }
   */
  const std::string stopcondition = [this] {
    switch (this->GetStopCondition())
    {

      case MaximumNumberOfIterations:
        return "Maximum number of iterations has been reached";
      case MetricError:
        return "Error in metric";
      default:
        return "Unknown";
    }
  }();

  /** Print the stopping condition */
  log::info(std::ostringstream{} << "Stopping condition: " << stopcondition << ".");

} // end AfterEachResolution()


/**
 * ******************* AfterRegistration ************************
 */

template <typename TElastix>
void
StandardGradientDescent<TElastix>::AfterRegistration()
{
  /** Print the best metric value */
  double bestValue = this->GetValue();
  log::info(std::ostringstream{} << '\n' << "Final metric value  = " << bestValue);

} // end AfterRegistration()


/**
 * ****************** StartOptimization *************************
 */

template <typename TElastix>
void
StandardGradientDescent<TElastix>::StartOptimization()
{
  /** Check if the entered scales are correct and != [ 1 1 1 ...] */
  this->SetUseScales(false);
  const ScalesType & scales = this->GetScales();
  if (scales.GetSize() == this->GetInitialPosition().GetSize())
  {
    ScalesType unit_scales(scales.GetSize(), 1.0);
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

template <typename TElastix>
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
