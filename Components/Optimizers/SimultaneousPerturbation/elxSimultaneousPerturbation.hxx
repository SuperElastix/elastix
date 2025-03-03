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

#ifndef elxSimultaneousPerturbation_hxx
#define elxSimultaneousPerturbation_hxx

#include "elxSimultaneousPerturbation.h"
#include <itkDeref.h>
#include <iomanip>
#include <string>
#include <vnl/vnl_math.h>

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template <typename TElastix>
SimultaneousPerturbation<TElastix>::SimultaneousPerturbation()
{
  this->m_ShowMetricValues = false;
} // end Constructor


/**
 * ***************** BeforeRegistration ***********************
 */

template <typename TElastix>
void
SimultaneousPerturbation<TElastix>::BeforeRegistration()
{
  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

  std::string showMetricValues("false");
  configuration.ReadParameter(showMetricValues, "ShowMetricValues", 0);
  if (showMetricValues == "false")
  {
    this->m_ShowMetricValues = false;
  }
  else
  {
    this->m_ShowMetricValues = true;
  }

  /** Add the target cell "stepsize" to IterationInfo.*/
  this->AddTargetCellToIterationInfo("2:Metric");
  this->AddTargetCellToIterationInfo("3:Gain a_k");
  this->AddTargetCellToIterationInfo("4:||Gradient||");

  /** Format the metric and stepsize as floats */
  this->GetIterationInfoAt("2:Metric") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("3:Gain a_k") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("4:||Gradient||") << std::showpoint << std::fixed;

} // end BeforeRegistration


/**
 * ***************** BeforeEachResolution ***********************
 */

template <typename TElastix>
void
SimultaneousPerturbation<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level.*/
  auto level = static_cast<unsigned int>(this->m_Registration->GetAsITKBaseType()->GetCurrentLevel());

  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

  /** Set the maximumNumberOfIterations.*/
  unsigned int maximumNumberOfIterations = 500;
  configuration.ReadParameter(
    maximumNumberOfIterations, "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0);
  this->SetMaximumNumberOfIterations(maximumNumberOfIterations);

  /** Set the number of perturbation used to construct a gradient estimate g_k. */
  unsigned int numberOfPerturbations = 1;
  configuration.ReadParameter(numberOfPerturbations, "NumberOfPerturbations", this->GetComponentLabel(), level, 0);
  this->SetNumberOfPerturbations(numberOfPerturbations);

  /** \todo call the GuessParameters function */
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

  this->Seta(a);
  this->Setc(c);
  this->SetA(A);
  this->SetAlpha(alpha);
  this->SetGamma(gamma);

  /** Ignore the build-in stop criterion; it's quite ad hoc. */
  this->SetTolerance(0.0);

} // end BeforeEachResolution


/**
 * ***************** AfterEachIteration *************************
 */

template <typename TElastix>
void
SimultaneousPerturbation<TElastix>::AfterEachIteration()
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
   * \todo You may also choose to select new samples upon every evaluation
   * of the metric value
   */
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
SimultaneousPerturbation<TElastix>::AfterEachResolution()
{

  /**
   * enum   StopConditionType {  MaximumNumberOfIterations, MetricError }
   * ignore the BelowTolerance-criterion.
   */
  const std::string stopcondition = [this] {
    switch (this->GetStopCondition())
    {

      case StopConditionSPSAOptimizerEnum::MaximumNumberOfIterations:
        return "Maximum number of iterations has been reached";
      case StopConditionSPSAOptimizerEnum::MetricError:
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
SimultaneousPerturbation<TElastix>::AfterRegistration()
{
  /** Print the best metric value */
  double bestValue = this->GetValue();
  log::info(std::ostringstream{} << '\n' << "Final metric value  = " << bestValue);

} // end AfterRegistration


/**
 * ******************* SetInitialPosition ***********************
 */

template <typename TElastix>
void
SimultaneousPerturbation<TElastix>::SetInitialPosition(const ParametersType & param)
{
  /** Override the implementation in itkOptimizer.h, to
   * ensure that the scales array and the parameters
   * array have the same size.
   */

  /** Call the Superclass' implementation. */
  this->Superclass1::SetInitialPosition(param);

  /** Set the scales array to the same size if the size has been changed */
  ScalesType   scales = this->GetScales();
  unsigned int paramsize = param.Size();

  if ((scales.Size()) != paramsize)
  {
    ScalesType newscales(paramsize, 1.0);
    this->SetScales(newscales);
  }

  /** \todo to optimizerbase? */

} // end SetInitialPosition


} // end namespace elastix

#endif // end #ifndef elxSimultaneousPerturbation_hxx
