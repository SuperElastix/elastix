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
#include <iomanip>
#include <string>
#include "math.h"

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template <class TElastix>
FiniteDifferenceGradientDescent<TElastix>::FiniteDifferenceGradientDescent()
{
  this->m_ShowMetricValues = false;
} // end Constructor


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void
FiniteDifferenceGradientDescent<TElastix>::BeforeRegistration()
{
  std::string showMetricValues("false");
  this->GetConfiguration()->ReadParameter(showMetricValues, "ShowMetricValues", 0);
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

template <class TElastix>
void
FiniteDifferenceGradientDescent<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level.*/
  unsigned int level = static_cast<unsigned int>(this->m_Registration->GetAsITKBaseType()->GetCurrentLevel());

  /** Set the maximumNumberOfIterations.*/
  unsigned int maximumNumberOfIterations = 500;
  this->m_Configuration->ReadParameter(
    maximumNumberOfIterations, "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0);
  this->SetNumberOfIterations(maximumNumberOfIterations);

  /** \todo  GuessParameters function */
  double a = 400.0;
  double c = 1.0;
  double A = 50.0;
  double alpha = 0.602;
  double gamma = 0.101;

  this->GetConfiguration()->ReadParameter(a, "SP_a", this->GetComponentLabel(), level, 0);
  this->GetConfiguration()->ReadParameter(c, "SP_c", this->GetComponentLabel(), level, 0);
  this->GetConfiguration()->ReadParameter(A, "SP_A", this->GetComponentLabel(), level, 0);
  this->GetConfiguration()->ReadParameter(alpha, "SP_alpha", this->GetComponentLabel(), level, 0);
  this->GetConfiguration()->ReadParameter(gamma, "SP_gamma", this->GetComponentLabel(), level, 0);

  this->SetParam_a(a);
  this->SetParam_c(c);
  this->SetParam_A(A);
  this->SetParam_alpha(alpha);
  this->SetParam_gamma(gamma);

} // end BeforeEachResolution


/**
 * ***************** AfterEachIteration *************************
 */

template <class TElastix>
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

template <class TElastix>
void
FiniteDifferenceGradientDescent<TElastix>::AfterEachResolution()
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

} // end AfterEachResolution


/**
 * ******************* AfterRegistration ************************
 */

template <class TElastix>
void
FiniteDifferenceGradientDescent<TElastix>::AfterRegistration()
{
  /** Print the best metric value */
  double bestValue;
  if (this->m_ShowMetricValues)
  {
    bestValue = this->GetValue();
    elxout << '\n' << "Final metric value  = " << bestValue << std::endl;
  }
  else
  {
    elxout << '\n'
           << "Run Elastix again with the option \"ShowMetricValues\" set to \"true\", to see information about the "
              "metric values. "
           << std::endl;
  }

} // end AfterRegistration


/**
 * ******************* StartOptimization ***********************
 */

template <class TElastix>
void
FiniteDifferenceGradientDescent<TElastix>::StartOptimization()
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

  this->Superclass1::StartOptimization();

} // end StartOptimization


} // end namespace elastix

#endif // end #ifndef elxFiniteDifferenceGradientDescent_hxx
