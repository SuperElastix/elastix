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

#ifndef elxCMAEvolutionStrategy_hxx
#define elxCMAEvolutionStrategy_hxx

#include "elxCMAEvolutionStrategy.h"
#include <iomanip>
#include <string>
#include <vnl/vnl_math.h>

namespace elastix
{

/**
 * ***************** StartOptimization ************************
 */

template <class TElastix>
void
CMAEvolutionStrategy<TElastix>::StartOptimization()
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

  /** Call the superclass */
  this->Superclass1::StartOptimization();

} // end StartOptimization


/**
 * ***************** InitializeProgressVariables ************************
 */

template <class TElastix>
void
CMAEvolutionStrategy<TElastix>::InitializeProgressVariables()
{
  this->Superclass1::InitializeProgressVariables();

  /** Print some settings that can be automatically determined by the optimizer. */
  elxout << "The CMAEvolutionStrategy optimizer uses the following settings:\n"
         << "PopulationSize = " << this->GetPopulationSize() << "\n"
         << "NumberOfParents = " << this->GetNumberOfParents() << "\n"
         << "UseCovarianceMatrixAdaptation = " << this->GetUseCovarianceMatrixAdaptation() << "\n"
         << "UpdateBDPeriod = " << this->GetUpdateBDPeriod() << "\n"
         << std::endl;

} // end InitializeProgressVariables


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void
CMAEvolutionStrategy<TElastix>::BeforeRegistration()
{
  /** Add target cells to xout[IterationInfo.*/
  this->AddTargetCellToIterationInfo("2:Metric");
  this->AddTargetCellToIterationInfo("3:StepLength");
  this->AddTargetCellToIterationInfo("4:||Step||");
  this->AddTargetCellToIterationInfo("5a:Sigma");
  this->AddTargetCellToIterationInfo("5b:MaximumD");
  this->AddTargetCellToIterationInfo("5c:MinimumD");

  /** Format the metric and stepsize as floats */
  this->GetIterationInfoAt("2:Metric") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("3:StepLength") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("4:||Step||") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("5a:Sigma") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("5b:MaximumD") << std::showpoint << std::fixed;
  this->GetIterationInfoAt("5c:MinimumD") << std::showpoint << std::fixed;

} // end BeforeRegistration


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
CMAEvolutionStrategy<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level.*/
  unsigned int level = static_cast<unsigned int>(this->m_Registration->GetAsITKBaseType()->GetCurrentLevel());

  /** Set MaximumNumberOfIterations.*/
  unsigned int maximumNumberOfIterations = 500;
  this->m_Configuration->ReadParameter(
    maximumNumberOfIterations, "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0);
  this->SetMaximumNumberOfIterations(maximumNumberOfIterations);

  /** Set the length of the initial step (InitialSigma). */
  double stepLength = 1.0;
  this->m_Configuration->ReadParameter(stepLength, "StepLength", this->GetComponentLabel(), level, 0);
  this->SetInitialSigma(stepLength);

  /** Set ValueTolerance */
  double valueTolerance = 0.00001;
  this->m_Configuration->ReadParameter(valueTolerance, "ValueTolerance", this->GetComponentLabel(), level, 0);
  this->SetValueTolerance(valueTolerance);

  /** Set PopulationSize */
  unsigned int populationSize = 0;
  this->m_Configuration->ReadParameter(populationSize, "PopulationSize", this->GetComponentLabel(), level, 0);
  this->SetPopulationSize(populationSize);

  /** Set NumberOfParents */
  unsigned int numberOfParents = 0;
  this->m_Configuration->ReadParameter(numberOfParents, "NumberOfParents", this->GetComponentLabel(), level, 0);
  this->SetNumberOfParents(numberOfParents);

  /** Set UseDecayingSigma */
  bool useDecayingSigma = false;
  this->m_Configuration->ReadParameter(useDecayingSigma, "UseDecayingSigma", this->GetComponentLabel(), level, 0);
  this->SetUseDecayingSigma(useDecayingSigma);

  /** Set SigmaDecayA */
  double sigmaDecayA = 50.0;
  this->m_Configuration->ReadParameter(sigmaDecayA, "SP_A", this->GetComponentLabel(), level, 0);
  this->SetSigmaDecayA(sigmaDecayA);

  /** Set SigmaDecayAlpha */
  double sigmaDecayAlpha = 0.602;
  this->m_Configuration->ReadParameter(sigmaDecayAlpha, "SP_alpha", this->GetComponentLabel(), level, 0);
  this->SetSigmaDecayAlpha(sigmaDecayAlpha);

  /** Set UseCovarianceMatrixAdaptation */
  bool useCovarianceMatrixAdaptation = true;
  this->m_Configuration->ReadParameter(
    useCovarianceMatrixAdaptation, "UseCovarianceMatrixAdaptation", this->GetComponentLabel(), level, 0);
  this->SetUseCovarianceMatrixAdaptation(useCovarianceMatrixAdaptation);

  /** Set RecombinationWeightsPreset */
  std::string recombinationWeightsPreset = "superlinear";
  this->m_Configuration->ReadParameter(
    recombinationWeightsPreset, "RecombinationWeightsPreset", this->GetComponentLabel(), level, 0);
  this->SetRecombinationWeightsPreset(recombinationWeightsPreset);

  /** Set UpdateBDPeriod */
  unsigned int updateBDPeriod = 0;
  this->m_Configuration->ReadParameter(updateBDPeriod, "UpdateBDPeriod", this->GetComponentLabel(), level, 0);
  this->SetUpdateBDPeriod(updateBDPeriod);

  /** Set PositionToleranceMin */
  double positionToleranceMin = 1e-8;
  this->m_Configuration->ReadParameter(
    positionToleranceMin, "PositionToleranceMin", this->GetComponentLabel(), level, 0);
  this->SetPositionToleranceMin(positionToleranceMin);

  /** Set PositionToleranceMax */
  double positionToleranceMax = 1e8;
  this->m_Configuration->ReadParameter(
    positionToleranceMax, "PositionToleranceMax", this->GetComponentLabel(), level, 0);
  this->SetPositionToleranceMax(positionToleranceMax);

  /** Set MaximumDeviation */
  double maximumDeviation = 10.0 * positionToleranceMax * stepLength;
  this->m_Configuration->ReadParameter(maximumDeviation, "MaximumDeviation", this->GetComponentLabel(), level, 0);
  this->SetMaximumDeviation(maximumDeviation);

  /** Set MinimumDeviation */
  double minimumDeviation = 0.0;
  this->m_Configuration->ReadParameter(minimumDeviation, "MinimumDeviation", this->GetComponentLabel(), level, 0);
  this->SetMinimumDeviation(minimumDeviation);

} // end BeforeEachResolution


/**
 * ***************** AfterEachIteration *************************
 */

template <class TElastix>
void
CMAEvolutionStrategy<TElastix>::AfterEachIteration()
{
  /** Print some information. */
  this->GetIterationInfoAt("2:Metric") << this->GetCurrentValue();
  this->GetIterationInfoAt("3:StepLength") << this->GetCurrentStepLength();
  this->GetIterationInfoAt("4:||Step||") << this->GetCurrentScaledStep().magnitude();
  this->GetIterationInfoAt("5a:Sigma") << this->GetCurrentSigma();
  this->GetIterationInfoAt("5b:MaximumD") << this->GetCurrentMaximumD();
  this->GetIterationInfoAt("5c:MinimumD") << this->GetCurrentMinimumD();

  /** Select new samples if desired. These
   * will be used in the next iteration */
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
CMAEvolutionStrategy<TElastix>::AfterEachResolution()
{
  /**
    enum StopConditionType {
    MetricError,
    MaximumNumberOfIterations,
    PositionToleranceMin,
    PositionToleranceMax,
    ValueTolerance,
    ZeroStepLength,
    Unknown };  */

  std::string stopcondition;

  switch (this->GetStopCondition())
  {
    case MetricError:
      stopcondition = "Error in metric";
      break;

    case MaximumNumberOfIterations:
      stopcondition = "Maximum number of iterations has been reached";
      break;

    case PositionToleranceMin:
      stopcondition = "The minimum step length condition has been reached";
      break;

    case PositionToleranceMax:
      stopcondition = "The maximum step length condition has been reached";
      break;

    case ValueTolerance:
      stopcondition = "Almost no decrease in function value anymore";
      break;

    case ZeroStepLength:
      stopcondition = "The step length is 0";
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
CMAEvolutionStrategy<TElastix>::AfterRegistration()
{
  /** Print the best metric value */

  double bestValue = this->GetCurrentValue();
  elxout << '\n' << "Final metric value  = " << bestValue << std::endl;

} // end AfterRegistration


} // end namespace elastix

#endif // end #ifndef elxCMAEvolutionStrategy_hxx
