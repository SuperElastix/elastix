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

#ifndef elxConjugateGradient_hxx
#define elxConjugateGradient_hxx

#include "elxConjugateGradient.h"
#include <iomanip>
#include <string>
#include <vnl/vnl_math.h>

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template <typename TElastix>
ConjugateGradient<TElastix>::ConjugateGradient()
{
  this->m_LineOptimizer = LineOptimizerType::New();
  this->SetLineSearchOptimizer(this->m_LineOptimizer);
  this->m_EventPasser = EventPassThroughType::New();
  this->m_EventPasser->SetCallbackFunction(this, &Self::InvokeIterationEvent);
  this->m_LineOptimizer->AddObserver(itk::IterationEvent(), this->m_EventPasser);
  this->m_LineOptimizer->AddObserver(itk::StartEvent(), this->m_EventPasser);

  this->m_SearchDirectionMagnitude = 0.0;
  this->m_StartLineSearch = false;
  this->m_GenerateLineSearchIterations = false;
  this->m_StopIfWolfeNotSatisfied = true;
  this->m_WolfeIsStopCondition = false;

} // end Constructor


/**
 * ***************** InvokeIterationEvent ************************
 */

template <typename TElastix>
void
ConjugateGradient<TElastix>::InvokeIterationEvent(const itk::EventObject & event)
{
  if (typeid(event) == typeid(itk::StartEvent))
  {
    this->m_StartLineSearch = true;
    this->m_SearchDirectionMagnitude = this->m_LineOptimizer->GetLineSearchDirection().magnitude();
  }
  else
  {
    this->m_StartLineSearch = false;
  }

  if (this->m_GenerateLineSearchIterations)
  {
    this->InvokeEvent(itk::IterationEvent());
  }

  this->m_StartLineSearch = false;
} // end InvokeIterationEvent


/**
 * ***************** StartOptimization ************************
 */

template <typename TElastix>
void
ConjugateGradient<TElastix>::StartOptimization()
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


/**
 * ***************** LineSearch ************************
 */

template <typename TElastix>
void
ConjugateGradient<TElastix>::LineSearch(const ParametersType searchDir,
                                        double &             step,
                                        ParametersType &     x,
                                        MeasureType &        f,
                                        DerivativeType &     g)
{
  /** Call the superclass's implementation and ignore a
   * LineSearchError. Just report the error and assume convergence. */
  try
  {
    this->Superclass1::LineSearch(searchDir, step, x, f, g);
  }
  catch (const itk::ExceptionObject & err)
  {
    if (this->GetLineSearchOptimizer() == nullptr)
    {
      throw;
    }
    else if (this->GetStopCondition() != StopConditionType::LineSearchError)
    {
      throw;
    }
    else
    {
      log::error(std::ostringstream{} << err << '\n' << "The error is ignored and convergence is assumed.");
      step = 0.0;
      x = this->GetScaledCurrentPosition();
      f = this->GetCurrentValue();
      g = this->GetCurrentGradient();
    }
  }
} // end LineSearch


/**
 * ***************** DeterminePhase *****************************
 *
 * This method gives only sensible output if it is called
 * during iterating
 */

template <typename TElastix>
std::string
ConjugateGradient<TElastix>::DeterminePhase() const
{

  if (this->GetInLineSearch())
  {
    return std::string("LineOptimizing");
  }

  return std::string("Main");

} // end DeterminePhase


/**
 * ***************** BeforeRegistration ***********************
 */

template <typename TElastix>
void
ConjugateGradient<TElastix>::BeforeRegistration()
{
  /** Add target cells to IterationInfo.*/
  this->AddTargetCellToIterationInfo("1a:SrchDirNr");
  this->AddTargetCellToIterationInfo("1b:LineItNr");
  this->AddTargetCellToIterationInfo("2:Metric");
  this->AddTargetCellToIterationInfo("3:StepLength");
  this->AddTargetCellToIterationInfo("4a:||Gradient||");
  this->AddTargetCellToIterationInfo("4b:||SearchDir||");
  this->AddTargetCellToIterationInfo("4c:DirGradient");
  this->AddTargetCellToIterationInfo("5:Phase");
  this->AddTargetCellToIterationInfo("6a:Wolfe1");
  this->AddTargetCellToIterationInfo("6b:Wolfe2");
  this->AddTargetCellToIterationInfo("7:LinSrchStopCondition");

  /** Format the metric and stepsize as floats */
  this->GetIterationInfoAt("2:Metric") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("3:StepLength") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("4a:||Gradient||") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("4b:||SearchDir||") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("4c:DirGradient") << std::showpoint << std::fixed;

  /** Check in the parameter file whether line search iterations should
   * be generated */
  this->m_GenerateLineSearchIterations = false; // bool
  std::string generateLineSearchIterations = "false";
  this->m_Configuration->ReadParameter(generateLineSearchIterations, "GenerateLineSearchIterations", 0);
  if (generateLineSearchIterations == "true")
  {
    this->m_GenerateLineSearchIterations = true;
  }

} // end BeforeRegistration


/**
 * ***************** BeforeEachResolution ***********************
 */

template <typename TElastix>
void
ConjugateGradient<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level.*/
  auto level = static_cast<unsigned int>(this->m_Registration->GetAsITKBaseType()->GetCurrentLevel());

  /** Set the maximumNumberOfIterations.*/
  unsigned int maximumNumberOfIterations = 100;
  this->m_Configuration->ReadParameter(
    maximumNumberOfIterations, "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0);
  this->SetMaximumNumberOfIterations(maximumNumberOfIterations);

  /** Set the maximumNumberOfIterations used for a line search.*/
  unsigned int maximumNumberOfLineSearchIterations = 20;
  this->m_Configuration->ReadParameter(
    maximumNumberOfLineSearchIterations, "MaximumNumberOfLineSearchIterations", this->GetComponentLabel(), level, 0);
  this->m_LineOptimizer->SetMaximumNumberOfIterations(maximumNumberOfLineSearchIterations);

  /** Set the length of the initial step, used to bracket the minimum. */
  double stepLength = 1.0;
  this->m_Configuration->ReadParameter(stepLength, "StepLength", this->GetComponentLabel(), level, 0);
  this->m_LineOptimizer->SetInitialStepLengthEstimate(stepLength);

  /** Set the LineSearchValueTolerance */
  double lineSearchValueTolerance = 0.0001;
  this->m_Configuration->ReadParameter(
    lineSearchValueTolerance, "LineSearchValueTolerance", this->GetComponentLabel(), level, 0);
  this->m_LineOptimizer->SetValueTolerance(lineSearchValueTolerance);

  /** Set the LineSearchGradientTolerance */
  double lineSearchGradientTolerance = 0.9;
  this->m_Configuration->ReadParameter(
    lineSearchGradientTolerance, "LineSearchGradientTolerance", this->GetComponentLabel(), level, 0);
  this->m_LineOptimizer->SetGradientTolerance(lineSearchGradientTolerance);

  /** Set the GradientMagnitudeTolerance */
  double gradientMagnitudeTolerance = 0.000001;
  this->m_Configuration->ReadParameter(
    gradientMagnitudeTolerance, "GradientMagnitudeTolerance", this->GetComponentLabel(), level, 0);
  this->SetGradientMagnitudeTolerance(gradientMagnitudeTolerance);

  /** Set the ValueTolerance */
  double valueTolerance = 0.00001;
  this->m_Configuration->ReadParameter(valueTolerance, "ValueTolerance", this->GetComponentLabel(), level, 0);
  this->SetValueTolerance(valueTolerance);

  /** Set the definition of beta */
  std::string betaDefinition = "DaiYuanHestenesStiefel";
  this->m_Configuration->ReadParameter(betaDefinition, "ConjugateGradientType", this->GetComponentLabel(), level, 0);
  this->SetBetaDefinition(betaDefinition);

  /** Just guess this one: */
  this->SetMaxNrOfItWithoutImprovement(10);

  /** Check whether to stop optimisation if Wolfe conditions are not satisfied. */
  this->m_StopIfWolfeNotSatisfied = true;
  std::string stopIfWolfeNotSatisfied = "true";
  this->m_Configuration->ReadParameter(
    stopIfWolfeNotSatisfied, "StopIfWolfeNotSatisfied", this->GetComponentLabel(), level, 0);
  if (stopIfWolfeNotSatisfied == "false")
  {
    this->m_StopIfWolfeNotSatisfied = false;
  }

  this->m_WolfeIsStopCondition = false;
  this->m_SearchDirectionMagnitude = 0.0;
  this->m_StartLineSearch = false;

} // end BeforeEachResolution


/**
 * ***************** AfterEachIteration *************************
 */

template <typename TElastix>
void
ConjugateGradient<TElastix>::AfterEachIteration()
{
  /** Print some information. */

  if (this->GetStartLineSearch())
  {
    this->GetIterationInfoAt("1b:LineItNr") << "start";
  }
  else
  {
    /**
     * If we are in a line search iteration the current line search
     * iteration number is printed.
     * If we are in a "main" iteration (no line search) the last
     * line search iteration number (so the number of line search
     * iterations minus one) is printed out.
     */
    this->GetIterationInfoAt("1b:LineItNr") << this->m_LineOptimizer->GetCurrentIteration();
  }

  if (this->GetInLineSearch())
  {
    this->GetIterationInfoAt("2:Metric") << this->m_LineOptimizer->GetCurrentValue();
    this->GetIterationInfoAt("3:StepLength") << this->m_LineOptimizer->GetCurrentStepLength();
    LineOptimizerType::DerivativeType cd;
    this->m_LineOptimizer->GetCurrentDerivative(cd);
    this->GetIterationInfoAt("4a:||Gradient||") << cd.magnitude();
    this->GetIterationInfoAt("7:LinSrchStopCondition") << "---";
  } // end if in line search
  else
  {
    this->GetIterationInfoAt("2:Metric") << this->GetCurrentValue();
    this->GetIterationInfoAt("3:StepLength") << this->GetCurrentStepLength();
    this->GetIterationInfoAt("4a:||Gradient||") << this->GetCurrentGradient().magnitude();
    this->GetIterationInfoAt("7:LinSrchStopCondition") << this->GetLineSearchStopCondition();
  } // end else (not in line search)

  this->GetIterationInfoAt("1a:SrchDirNr") << this->GetCurrentIteration();
  this->GetIterationInfoAt("5:Phase") << this->DeterminePhase();
  this->GetIterationInfoAt("4b:||SearchDir||") << this->m_SearchDirectionMagnitude;
  this->GetIterationInfoAt("4c:DirGradient") << this->m_LineOptimizer->GetCurrentDirectionalDerivative();
  if (this->m_LineOptimizer->GetSufficientDecreaseConditionSatisfied())
  {
    this->GetIterationInfoAt("6a:Wolfe1") << "true";
  }
  else
  {
    this->GetIterationInfoAt("6a:Wolfe1") << "false";
  }
  if (this->m_LineOptimizer->GetCurvatureConditionSatisfied())
  {
    this->GetIterationInfoAt("6b:Wolfe2") << "true";
  }
  else
  {
    this->GetIterationInfoAt("6b:Wolfe2") << "false";
  }

  if (!(this->GetInLineSearch()))
  {
    /** Set the initial step length estimate for the next line search
     * to the result of the last iteration */
    this->m_LineOptimizer->SetInitialStepLengthEstimate(this->GetCurrentStepLength());

    /** If new samples: compute a new gradient and value. These
     * will be used in the computation of a new search direction */
    if (this->GetNewSamplesEveryIteration())
    {
      this->SelectNewSamples();
      try
      {
        this->GetScaledValueAndDerivative(
          this->GetScaledCurrentPosition(), this->m_CurrentValue, this->m_CurrentGradient);
      }
      catch (const itk::ExceptionObject &)
      {
        this->m_StopCondition = StopConditionType::MetricError;
        this->StopOptimization();
        throw;
      }
    } // end if new samples every iteration
  } // end if not in line search

} // end AfterEachIteration


/**
 * ***************** AfterEachResolution *************************
 */

template <typename TElastix>
void
ConjugateGradient<TElastix>::AfterEachResolution()
{
  /**
  enum {
    MetricError,
    LineSearchError,
    MaximumNumberOfIterations,
    GradientMagnitudeTolerance,
    ValueTolerance,
    Unknown }
    */

  std::string stopcondition;

  if (this->m_WolfeIsStopCondition)
  {
    stopcondition = "Wolfe conditions are not satisfied";
  }
  else
  {
    stopcondition = [this] {
      switch (this->GetStopCondition())
      {

        case StopConditionType::MetricError:
          return "Error in metric";
        case StopConditionType::LineSearchError:
          return "Error in LineSearch";
        case StopConditionType::MaximumNumberOfIterations:
          return "Maximum number of iterations has been reached";
        case StopConditionType::GradientMagnitudeTolerance:
          return "The gradient magnitude has (nearly) vanished";
        case StopConditionType::ValueTolerance:
          return "Almost no decrease in function value anymore";
        case StopConditionType::InfiniteBeta:
          return "The beta factor became infinite";
        default:
          return "Unknown";
      }
    }();
  } // end else

  /** Print the stopping condition */
  log::info(std::ostringstream{} << "Stopping condition: " << stopcondition << ".");

} // end AfterEachResolution


/**
 * ******************* AfterRegistration ************************
 */

template <typename TElastix>
void
ConjugateGradient<TElastix>::AfterRegistration()
{
  /** Print the best metric value */

  double bestValue = this->GetCurrentValue();
  log::info(std::ostringstream{} << '\n' << "Final metric value  = " << bestValue);

} // end AfterRegistration


/**
 * *********************** TestConvergence *****************
 */

template <typename TElastix>
bool
ConjugateGradient<TElastix>::TestConvergence(bool firstLineSearchDone)
{
  bool convergence = this->Superclass1::TestConvergence(firstLineSearchDone);

  /** Stop if the Wolfe conditions are not satisfied
   * NB: this check is only done when 'convergence' wasn't true already */
  if (this->m_StopIfWolfeNotSatisfied && !convergence && firstLineSearchDone)
  {
    if ((!(this->m_LineOptimizer->GetCurvatureConditionSatisfied())) ||
        (!(this->m_LineOptimizer->GetSufficientDecreaseConditionSatisfied())))
    {
      /** Stop the optimisation */
      this->m_WolfeIsStopCondition = true;
      convergence = true;
    }
  }

  return convergence;

} // end TestConvergence


/**
 * ***************** GetLineSearchStopCondition *****************
 */

template <typename TElastix>
std::string
ConjugateGradient<TElastix>::GetLineSearchStopCondition() const
{
  /** Must be repeated here; otherwise the StopconditionTypes of the
   * GenericConjugateGradientOptimizer and the LineSearchOptimizer
   * are mixed up. */
  enum class LineSearchStopConditionType
  {
    StrongWolfeConditionsSatisfied,
    MetricError,
    MaximumNumberOfIterations,
    StepTooSmall,
    StepTooLarge,
    IntervalTooSmall,
    RoundingError,
    AscentSearchDirection,
    Unknown
  };

  switch (static_cast<LineSearchStopConditionType>(this->m_LineOptimizer->GetStopCondition()))
  {

    case LineSearchStopConditionType::StrongWolfeConditionsSatisfied:
      return "WolfeSatisfied";
    case LineSearchStopConditionType::MetricError:
      return "MetricError";
    case LineSearchStopConditionType::MaximumNumberOfIterations:
      return "MaxNrIterations";
    case LineSearchStopConditionType::StepTooSmall:
      return "StepTooSmall";
    case LineSearchStopConditionType::StepTooLarge:
      return "StepTooLarge";
    case LineSearchStopConditionType::IntervalTooSmall:
      return "IntervalTooSmall";
    case LineSearchStopConditionType::RoundingError:
      return "RoundingError";
    case LineSearchStopConditionType::AscentSearchDirection:
      return "AscentSearchDir";
    default:
      return "Unknown";
  }

} // end GetLineSearchStopCondition


} // end namespace elastix

#endif // end #ifndef elxConjugateGradient_hxx
