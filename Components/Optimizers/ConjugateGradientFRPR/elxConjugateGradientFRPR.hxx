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

#ifndef elxConjugateGradientFRPR_hxx
#define elxConjugateGradientFRPR_hxx

#include "elxConjugateGradientFRPR.h"
#include <cmath>
#include <iomanip>
#include <string>
#include <vnl/vnl_math.h>

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template <class TElastix>
ConjugateGradientFRPR<TElastix>::ConjugateGradientFRPR()
{
  this->m_LineBracketing = false;
  this->m_LineOptimizing = false;
  this->m_CurrentStepLength = 0.0;
  this->m_CurrentSearchDirectionMagnitude = 0.0;
  this->m_CurrentDerivativeMagnitude = 0.0;

} // end Constructor


/**
 * ***************** DeterminePhase *****************************
 *
 * This method gives only sensible output if it is called
 * during iterating
 */

template <class TElastix>
const char *
ConjugateGradientFRPR<TElastix>::DeterminePhase() const
{
  if (this->GetLineBracketing())
  {
    return "LineBracketing";
  }

  if (this->GetLineOptimizing())
  {
    return "LineOptimizing";
  }

  return "Main";

} // end DeterminePhase


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void
ConjugateGradientFRPR<TElastix>::BeforeRegistration()
{

  /** Add target cells to IterationInfo.*/
  this->AddTargetCellToIterationInfo("1a:SrchDirNr");
  this->AddTargetCellToIterationInfo("1b:LineItNr");
  this->AddTargetCellToIterationInfo("2:Metric");
  this->AddTargetCellToIterationInfo("3:StepLength");
  this->AddTargetCellToIterationInfo("4a:||Gradient||");
  this->AddTargetCellToIterationInfo("4b:||SearchDir||");
  this->AddTargetCellToIterationInfo("5:Phase");

  /** Format some fields as floats */
  this->GetIterationInfoAt("2:Metric") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("3:StepLength") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("4a:||Gradient||") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("4b:||SearchDir||") << std::showpoint << std::fixed;

  /** \todo: call the correct functions */

} // end BeforeRegistration


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
ConjugateGradientFRPR<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level.*/
  unsigned int level = static_cast<unsigned int>(this->m_Registration->GetAsITKBaseType()->GetCurrentLevel());

  /** Set the maximumNumberOfIterations.*/
  unsigned int maximumNumberOfIterations = 100;
  this->m_Configuration->ReadParameter(
    maximumNumberOfIterations, "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0);
  this->SetMaximumIteration(maximumNumberOfIterations);

  /** Set the maximumNumberOfIterations used for a line search.*/
  unsigned int maximumNumberOfLineSearchIterations = 20;
  this->m_Configuration->ReadParameter(
    maximumNumberOfLineSearchIterations, "MaximumNumberOfLineSearchIterations", this->GetComponentLabel(), level, 0);
  this->SetMaximumLineIteration(maximumNumberOfLineSearchIterations);

  /** Set the length of the initial step, used to bracket the minimum. */
  double stepLength = 1.0;
  this->m_Configuration->ReadParameter(stepLength, "StepLength", this->GetComponentLabel(), level, 0);
  this->SetStepLength(stepLength);

  /** Set the ValueTolerance; convergence is declared if:
   * 2.0 * abs(f2 - f1) <=  ValueTolerance * (abs(f2) + std::abs(f1))
   */
  double valueTolerance = 0.00001;
  this->m_Configuration->ReadParameter(valueTolerance, "ValueTolerance", this->GetComponentLabel(), level, 0);
  this->SetValueTolerance(valueTolerance);

  /** Set the LineSearchStepTolerance; convergence of the line search is declared if:
   *
   * abs(x-xm) <= tol * abs(x) - (b-a)/2
   * where:
   * x = current mininum of the gain
   * a, b = current brackets around the minimum
   * xm = (a+b)/2
   */
  double stepTolerance = 0.00001;
  this->m_Configuration->ReadParameter(stepTolerance, "LineSearchStepTolerance", this->GetComponentLabel(), level, 0);
  this->SetStepTolerance(stepTolerance);

} // end BeforeEachResolution


/**
 * ***************** AfterEachIteration *************************
 */

template <class TElastix>
void
ConjugateGradientFRPR<TElastix>::AfterEachIteration()
{

  /** Print some information. */
  this->GetIterationInfoAt("1a:SrchDirNr") << this->GetCurrentIteration();
  this->GetIterationInfoAt("1b:LineItNr") << this->GetCurrentLineIteration();
  this->GetIterationInfoAt("2:Metric") << this->GetValue();
  this->GetIterationInfoAt("4b:||SearchDir||") << this->GetCurrentSearchDirectionMagnitude();
  this->GetIterationInfoAt("5:Phase") << this->DeterminePhase();

  /** If main iteration (NewDirection) */
  if ((!(this->GetLineBracketing())) && (!(this->GetLineOptimizing())))
  {
    this->GetIterationInfoAt("3:StepLength") << this->GetCurrentStepLength();
    this->GetIterationInfoAt("4a:||Gradient||") << this->GetCurrentDerivativeMagnitude();
  }
  else
  {
    if (this->GetLineBracketing())
    {
      this->GetIterationInfoAt("3:StepLength") << this->GetCurrentStepLength();
    }
    else
    {
      /** No step length known now */
      this->GetIterationInfoAt("3:StepLength") << "---";
    }
    this->GetIterationInfoAt("4a:||Gradient||") << "---";
  } // end if main iteration

} // end AfterEachIteration


/**
 * ***************** AfterEachResolution *************************
 */

template <class TElastix>
void
ConjugateGradientFRPR<TElastix>::AfterEachResolution()
{

  /**
   * enum   StopConditionType {  MaximumNumberOfIterations, MetricError }
   */

  /** \todo StopConditions are not yet implemented in the FRPROptimizer;
   * Uncomment the following code when they are implemented.

  std::string stopcondition;

  switch( this->GetStopCondition() )
  {

  case MaximumNumberOfIterations :
    stopcondition = "Maximum number of iterations has been reached";
    break;

  case MetricError :
    stopcondition = "Error in metric";
    break;

  default:
    stopcondition = "Unknown";
    break;

  }
*/
  /** Print the stopping condition */
  // elxout << "Stopping condition: " << stopcondition << "." << std::endl;

} // end AfterEachResolution


/**
 * ******************* AfterRegistration ************************
 */

template <class TElastix>
void
ConjugateGradientFRPR<TElastix>::AfterRegistration()
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
ConjugateGradientFRPR<TElastix>::SetInitialPosition(const ParametersType & param)
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


/**
 * *************** GetValueAndDerivative ***********************
 */

template <class TElastix>
void
ConjugateGradientFRPR<TElastix>::GetValueAndDerivative(ParametersType p, double * val, ParametersType * xi)
{
  /** This implementation forces the metric to select new samples
   * (if the user asked for this), calls the Superclass'
   * implementation and caches the computed derivative. */

  /** Select new spatial samples for the computation of the metric */
  if (this->GetNewSamplesEveryIteration())
  {
    this->SelectNewSamples();
  }

  this->Superclass1::GetValueAndDerivative(p, val, xi);
  this->m_CurrentDerivativeMagnitude = xi->magnitude();

} // end GetValueAndDerivative


/**
 * *************** LineBracket ****************************
 */

template <class TElastix>
void
ConjugateGradientFRPR<TElastix>::LineBracket(double * ax,
                                             double * bx,
                                             double * cx,
                                             double * fa,
                                             double * fb,
                                             double * fc)
{
  /** This implementation sets the LineBracketing flag to 'true', calls the
   * superclass' implementation, remembers the current step length (bx), invokes
   * an iteration event, and sets the LineBracketing flag to 'false' */

  this->SetLineBracketing(true);
  this->Superclass1::LineBracket(ax, bx, cx, fa, fb, fc);
  this->m_CurrentStepLength = *bx;
  this->InvokeEvent(itk::IterationEvent());
  this->SetLineBracketing(false);

} // end LineBracket


/**
 * *************** BracketedLineOptimize *********************
 */

template <class TElastix>
void
ConjugateGradientFRPR<TElastix>::BracketedLineOptimize(double   ax,
                                                       double   bx,
                                                       double   cx,
                                                       double   fa,
                                                       double   fb,
                                                       double   fc,
                                                       double * extX,
                                                       double * extVal)
{
  /** This implementation sets the LineOptimizing flag to 'true', calls the
   * the superclass's implementation, remembers the resulting step length,
   * and sets the LineOptimizing flag to 'false' again. */

  this->SetLineOptimizing(true);
  this->Superclass1::BracketedLineOptimize(ax, bx, cx, fa, fb, fc, extX, extVal);
  this->m_CurrentStepLength = *extX;
  this->SetLineOptimizing(false);

} // end BracketedLineOptimize


/**
 * ************************** LineOptimize ********************************
 * store the line search direction's (xi) magnitude and call the superclass'
 * implementation.
 */
template <class TElastix>
void
ConjugateGradientFRPR<TElastix>::LineOptimize(ParametersType * p, ParametersType xi, double * val)
{
  this->m_CurrentSearchDirectionMagnitude = xi.magnitude();
  this->Superclass1::LineOptimize(p, xi, val);
} // end LineOptimize


} // end namespace elastix

#endif // end #ifndef elxConjugateGradientFRPR_hxx
