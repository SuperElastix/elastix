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

#ifndef elxFiniteDifferenceGradientDescent_hxx
#define elxFiniteDifferenceGradientDescent_hxx

#include "elxFiniteDifferenceGradientDescent.h"
#include <itkDeref.h>
#include <iomanip>
#include <string>
#include "math.h"

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template <typename TElastix>
FiniteDifferenceGradientDescent<TElastix>::FiniteDifferenceGradientDescent()
{
  this->m_ShowMetricValues = false;
} // end Constructor


/**
 * ***************** BeforeRegistration ***********************
 */

template <typename TElastix>
void
FiniteDifferenceGradientDescent<TElastix>::BeforeRegistration()
{
  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

  std::string showMetricValues("false");
  configuration.ReadParameter(showMetricValues, "ShowMetricValues", 0);
  if (showMetricValues == "false")
  {
    this->m_ShowMetricValues = false;
    this->SetComputeCurrentValue(this->m_ShowMetricValues);
  }
  else
  {
    this->m_ShowMetricValues = true;
    this->SetComputeCurrentValue(this->m_ShowMetricValues);
  }

  /** Add some target cells to IterationInfo.*/
  this->AddTargetCellToIterationInfo("2:Metric");
  this->AddTargetCellToIterationInfo("3:Gain a_k");
  this->AddTargetCellToIterationInfo("4:||Gradient||");

  /** Format them as floats */
  this->GetIterationInfoAt("2:Metric") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("3:Gain a_k") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("4:||Gradient||") << std::showpoint << std::fixed;

} // end BeforeRegistration


/**
 * ***************** BeforeEachResolution ***********************
 */

template <typename TElastix>
void
FiniteDifferenceGradientDescent<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level.*/
  auto level = static_cast<unsigned int>(this->m_Registration->GetAsITKBaseType()->GetCurrentLevel());

  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

  /** Set the maximumNumberOfIterations.*/
  unsigned int maximumNumberOfIterations = 500;
  configuration.ReadParameter(
    maximumNumberOfIterations, "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0);
  this->SetNumberOfIterations(maximumNumberOfIterations);

  /** \todo  GuessParameters function */
  double a = 400.0;
  double c = 1.0;
  double A = 50.0;
  double alpha = 0.602;
  double gamma = 0.101;

  configuration.ReadParameter(a, "SP_a", this->GetComponentLabel(), level, 0);
  configuration.ReadParameter(c, "SP_c", this->GetComponentLabel(), level, 0);
  configuration.ReadParameter(A, "SP_A", this->GetComponentLabel(), level, 0);
  configuration.ReadParameter(alpha, "SP_alpha", this->GetComponentLabel(), level, 0);
  configuration.ReadParameter(gamma, "SP_gamma", this->GetComponentLabel(), level, 0);

  this->SetParam_a(a);
  this->SetParam_c(c);
  this->SetParam_A(A);
  this->SetParam_alpha(alpha);
  this->SetParam_gamma(gamma);

} // end BeforeEachResolution


/**
 * ***************** AfterEachIteration *************************
 */

template <typename TElastix>
void
FiniteDifferenceGradientDescent<TElastix>::AfterEachIteration()
{
  /** Print some information */

  if (this->m_ShowMetricValues)
  {
    this->GetIterationInfoAt("2:Metric") << this->GetValue();
  }
  else
  {
    this->GetIterationInfoAt("2:Metric") << "---";
  }
  this->GetIterationInfoAt("3:Gain a_k") << this->GetLearningRate();
  this->GetIterationInfoAt("4:||Gradient||") << this->GetGradientMagnitude();

  /** Select new spatial samples for the computation of the metric
   * \todo You may also choose to select new samples after evaluation
   * of the metric value */
  if (this->GetNewSamplesEveryIteration())
  {
    this->SelectNewSamples();
  }

} // end AfterEachIteration


/**
 * ***************** AfterEachResolution *************************
 */

template <typename TElastix>
void
FiniteDifferenceGradientDescent<TElastix>::AfterEachResolution()
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

} // end AfterEachResolution


/**
 * ******************* AfterRegistration ************************
 */

template <typename TElastix>
void
FiniteDifferenceGradientDescent<TElastix>::AfterRegistration()
{
  /** Print the best metric value */
  double bestValue;
  if (this->m_ShowMetricValues)
  {
    bestValue = this->GetValue();
    log::info(std::ostringstream{} << '\n' << "Final metric value  = " << bestValue);
  }
  else
  {
    log::info(std::ostringstream{}
              << '\n'
              << "Run Elastix again with the option \"ShowMetricValues\" set to \"true\", to see information about the "
                 "metric values. ");
  }

} // end AfterRegistration


/**
 * ******************* StartOptimization ***********************
 */

template <typename TElastix>
void
FiniteDifferenceGradientDescent<TElastix>::StartOptimization()
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

  this->Superclass1::StartOptimization();

} // end StartOptimization


} // end namespace elastix

#endif // end #ifndef elxFiniteDifferenceGradientDescent_hxx
