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

#ifndef elxRegularStepGradientDescent_hxx
#define elxRegularStepGradientDescent_hxx

#include "elxRegularStepGradientDescent.h"
#include <iomanip>
#include <string>
#include <vnl/vnl_math.h>

namespace elastix
{

/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void
RegularStepGradientDescent<TElastix>::BeforeRegistration()
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

template <class TElastix>
void
RegularStepGradientDescent<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level.*/
  unsigned int level = static_cast<unsigned int>(this->m_Registration->GetAsITKBaseType()->GetCurrentLevel());

  /** Set the Gradient Magnitude Stopping Criterion.*/
  double minGradientMagnitude = 1e-8;
  this->m_Configuration->ReadParameter(
    minGradientMagnitude, "MinimumGradientMagnitude", this->GetComponentLabel(), level, 0);
  this->SetGradientMagnitudeTolerance(minGradientMagnitude);

  /** Set the MaximumStepLength.*/
  double maxStepLength = 16.0 / std::pow(2.0, static_cast<int>(level));
  this->m_Configuration->ReadParameter(maxStepLength, "MaximumStepLength", this->GetComponentLabel(), level, 0);
  this->SetMaximumStepLength(maxStepLength);

  /** Set the MinimumStepLength.*/
  double minStepLength = 0.5 / std::pow(2.0, static_cast<int>(level));
  this->m_Configuration->ReadParameter(minStepLength, "MinimumStepLength", this->GetComponentLabel(), level, 0);
  this->SetMinimumStepLength(minStepLength);

  /** Set the Relaxation factor
   */
  double relaxationFactor = 0.5;
  this->m_Configuration->ReadParameter(relaxationFactor, "RelaxationFactor", this->GetComponentLabel(), level, 0);
  this->SetRelaxationFactor(relaxationFactor);

  /** \todo max and min steplength should maybe depend on the imagespacing or on something else... */

  /** Set the maximumNumberOfIterations.*/
  unsigned int maximumNumberOfIterations = 500;
  this->m_Configuration->ReadParameter(
    maximumNumberOfIterations, "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0);
  this->SetNumberOfIterations(maximumNumberOfIterations);

} // end BeforeEachResolution


/**
 * ***************** AfterEachIteration *************************
 */

template <class TElastix>
void
RegularStepGradientDescent<TElastix>::AfterEachIteration()
{
  /** Print some information */
  this->GetIterationInfoAt("2:Metric") << this->GetValue();
  this->GetIterationInfoAt("3:StepSize") << this->GetCurrentStepLength();
  this->GetIterationInfoAt("4:||Gradient||") << this->GetGradient().magnitude();
} // end AfterEachIteration


/**
 * ***************** AfterEachResolution *************************
 */

template <class TElastix>
void
RegularStepGradientDescent<TElastix>::AfterEachResolution()
{

  /**
   * enum   StopConditionType {   GradientMagnitudeTolerance = 1, StepTooSmall,
   * ImageNotAvailable, CostFunctionError, MaximumNumberOfIterations
   */
  std::string stopcondition;

  switch (this->GetStopCondition())
  {

    case StopConditionEnum::GradientMagnitudeTolerance:
      stopcondition = "Minimum gradient magnitude has been reached";
      break;

    case StopConditionEnum::StepTooSmall:
      stopcondition = "Minimum step size has been reached";
      break;

    case StopConditionEnum::MaximumNumberOfIterations:
      stopcondition = "Maximum number of iterations has been reached";
      break;

    case StopConditionEnum::ImageNotAvailable:
      stopcondition = "No image available";
      break;

    case StopConditionEnum::CostFunctionError:
      stopcondition = "Error in cost function";
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
RegularStepGradientDescent<TElastix>::AfterRegistration()
{
  /** Print the best metric value */
  double bestValue = this->GetValue();
  elxout << '\n' << "Final metric value  = " << bestValue << std::endl;

} // end AfterRegistration


/**
 * ******************* SetInitialPosition ***********************
 */

template <class TElastix>
void
RegularStepGradientDescent<TElastix>::SetInitialPosition(const ParametersType & param)
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
    ScalesType newscales(paramsize);
    newscales.Fill(1.0);
    this->SetScales(newscales);
  }

  /** \todo to optimizerbase? */

} // end SetInitialPosition


} // end namespace elastix

#endif // end #ifndef elxRegularStepGradientDescent_hxx
