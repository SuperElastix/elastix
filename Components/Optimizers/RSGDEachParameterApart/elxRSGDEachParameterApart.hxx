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

#ifndef elxRSGDEachParameterApart_hxx
#define elxRSGDEachParameterApart_hxx

#include "elxRSGDEachParameterApart.h"
#include <iomanip>
#include <string>
#include <vnl/vnl_math.h>

namespace elastix
{

/**
 * ***************** BeforeRegistration ***********************
 */

template <typename TElastix>
void
RSGDEachParameterApart<TElastix>::BeforeRegistration()
{
  /** Add the target cell "stepsize" to IterationInfo.*/
  this->AddTargetCellToIterationInfo("2:Metric");
  this->AddTargetCellToIterationInfo("3:StepSize");
  this->AddTargetCellToIterationInfo("4:||Gradient||");

  /** Format the metric and stepsize as floats */
  this->GetIterationInfoAt("2:Metric") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("3:StepSize") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("4:||Gradient||") << std::showpoint << std::fixed;

} // end BeforeRegistration


/**
 * ***************** BeforeEachResolution ***********************
 */

template <typename TElastix>
void
RSGDEachParameterApart<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level.*/
  auto level = static_cast<unsigned int>(this->m_Registration->GetAsITKBaseType()->GetCurrentLevel());

  /** Set the Gradient Magnitude Stopping Criterion.*/
  double minGradientMagnitude = 1e-8;
  this->m_Configuration->ReadParameter(
    minGradientMagnitude, "MinimumGradientMagnitude", this->GetComponentLabel(), level, 0);
  this->SetGradientMagnitudeTolerance(minGradientMagnitude);

  /** Set the MaximumStepLength.*/
  double maxStepLength = 16.0 / pow(2.0, static_cast<int>(level));
  this->m_Configuration->ReadParameter(maxStepLength, "MaximumStepLength", this->GetComponentLabel(), level, 0);
  this->SetMaximumStepLength(maxStepLength);

  /** Set the MinimumStepLength.*/
  double minStepLength = 0.5 / pow(2.0, static_cast<int>(level));
  this->m_Configuration->ReadParameter(minStepLength, "MinimumStepLength", this->GetComponentLabel(), level, 0);
  this->SetMinimumStepLength(minStepLength);

  /** Set the Relaxation factor
   * \todo Implement this also in the itkRSGDEachParameterApartOptimizer
   *  (just like in the RegularStepGradientDescentOptimizer) and
   * uncomment the following:
   */
  // double relaxationFactor = 0.5;
  // this->m_Configuration->ReadParameter( relaxationFactor,
  //  "RelaxationFactor", this->GetComponentLabel(), level, 0 );
  // this->SetRelaxationFactor( relaxationFactor );

  /** \todo max and min steplength should maybe depend on the imagespacing or on something else... */

  /** Set the maximumNumberOfIterations.*/
  unsigned int maximumNumberOfIterations = 100;
  this->m_Configuration->ReadParameter(
    maximumNumberOfIterations, "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0);
  this->SetNumberOfIterations(maximumNumberOfIterations);

} // end BeforeEachResolution


/**
 * ***************** AfterEachIteration *************************
 */

template <typename TElastix>
void
RSGDEachParameterApart<TElastix>::AfterEachIteration()
{
  /** Print some information */
  this->GetIterationInfoAt("2:Metric") << this->GetValue();
  this->GetIterationInfoAt("3:StepSize") << this->GetCurrentStepLength();
  this->GetIterationInfoAt("4:||Gradient||") << this->GetGradientMagnitude();

} // end AfterEachIteration


/**
 * ***************** AfterEachResolution *************************
 */

template <typename TElastix>
void
RSGDEachParameterApart<TElastix>::AfterEachResolution()
{

  /**
   * enum   StopConditionType {   GradientMagnitudeTolerance = 1, StepTooSmall,
   * ImageNotAvailable, SamplesNotAvailable, MaximumNumberOfIterations, MetricError
   */
  const std::string stopcondition = [this] {
    switch (this->GetStopCondition())
    {

      case GradientMagnitudeTolerance:
        return "Minimum gradient magnitude has been reached";
      case StepTooSmall:
        return "Minimum step size has been reached";
      case MaximumNumberOfIterations:
        return "Maximum number of iterations has been reached";
      case ImageNotAvailable:
        return "No image available";
      case SamplesNotAvailable:
        return "No samples available";
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
RSGDEachParameterApart<TElastix>::AfterRegistration()
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
RSGDEachParameterApart<TElastix>::SetInitialPosition(const ParametersType & param)
{
  /** Override the implementation in itkOptimizer.h, to
   * ensure that the scales array and the parameters
   * array have the same size.
   */

  /** Call the Superclass' implementation. */
  this->Superclass1::SetInitialPosition(param);

  /** Set the scales array to the same size if the size has been changed */
  if (const std::size_t paramsize{ param.Size() }; this->GetScales().Size() != paramsize)
  {
    this->SetScales(ScalesType(paramsize, 1.0));
  }

  /** \todo to optimizerbase? */

} // end SetInitialPosition


} // end namespace elastix

#endif // end #ifndef elxRSGDEachParameterApart_hxx
