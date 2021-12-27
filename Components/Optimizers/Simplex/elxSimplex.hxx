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

#ifndef elxSimplex_hxx
#define elxSimplex_hxx

#include "elxSimplex.h"
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
Simplex<TElastix>::BeforeRegistration()
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
Simplex<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level.*/
  unsigned int level = static_cast<unsigned int>(this->m_Registration->GetAsITKBaseType()->GetCurrentLevel());

  /** Set the value tolerance.*/
  double valueTolerance = 1e-8;
  this->m_Configuration->ReadParameter(valueTolerance, "ValueTolerance", this->GetComponentLabel(), level, 0);
  this->SetFunctionConvergenceTolerance(valueTolerance);

  /** Set the maximumNumberOfIterations.*/
  unsigned int maximumNumberOfIterations = 500;
  this->m_Configuration->ReadParameter(
    maximumNumberOfIterations, "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0);
  this->SetMaximumNumberOfIterations(maximumNumberOfIterations);

  /** Set the automaticinitialsimplex.*/
  bool automaticinitialsimplex = false;
  this->m_Configuration->ReadParameter(
    automaticinitialsimplex, "AutomaticInitialSimplex", this->GetComponentLabel(), level, 0);
  this->SetAutomaticInitialSimplex(automaticinitialsimplex);

  /** If no automaticinitialsimplex, InitialSimplexDelta should be given.*/
  if (!automaticinitialsimplex)
  {
    unsigned int numberofparameters =
      this->m_Elastix->GetElxTransformBase()->GetAsITKBaseType()->GetNumberOfParameters();
    ParametersType initialsimplexdelta(numberofparameters);
    initialsimplexdelta.Fill(1);

    for (unsigned int i = 0; i < numberofparameters; ++i)
    {
      this->m_Configuration->ReadParameter(initialsimplexdelta[i], "InitialSimplexDelta", i);
    }

    this->SetInitialSimplexDelta(initialsimplexdelta);
  }

} // end BeforeEachResolution()


/**
 * ***************** AfterEachIteration *************************
 */

template <class TElastix>
void
Simplex<TElastix>::AfterEachIteration()
{
  /** Print some information */
  this->GetIterationInfoAt("2:Metric") << this->GetCachedValue();
  // this->GetIterationInfoAt("3:StepSize") << this->GetStepLength();

} // end AfterEachIteration()


/**
 * ***************** AfterEachResolution *************************
 */

template <class TElastix>
void
Simplex<TElastix>::AfterEachResolution()
{
  /**
   * enum   StopConditionType {   GradientMagnitudeTolerance = 1, StepTooSmall,
   * ImageNotAvailable, CostFunctionError, MaximumNumberOfIterations
   */
  std::string stopcondition = this->GetStopConditionDescription();

  /** Print the stopping condition */
  elxout << "Stopping condition: " << stopcondition << "." << std::endl;

} // end AfterEachResolution()


/**
 * ******************* AfterRegistration ************************
 */

template <class TElastix>
void
Simplex<TElastix>::AfterRegistration()
{
  /** Print the best metric value */
  // double bestValue = this->GetValue();
  double bestValue = this->GetCachedValue();
  elxout << '\n' << "Final metric value  = " << bestValue << std::endl;

} // end AfterRegistration()


/**
 * ******************* SetInitialPosition ***********************
 */

template <class TElastix>
void
Simplex<TElastix>::SetInitialPosition(const ParametersType & param)
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

} // end SetInitialPosition()


} // end namespace elastix

#endif // end #ifndef elxSimplex_hxx
