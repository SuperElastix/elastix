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
#ifndef elxTransformRigidityPenaltyTerm_hxx
#define elxTransformRigidityPenaltyTerm_hxx

#include "elxTransformRigidityPenaltyTerm.h"

#include "itkChangeInformationImageFilter.h"
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
void
TransformRigidityPenalty<TElastix>::BeforeRegistration()
{
  /** Read the fixed rigidity image if desired. */
  std::string fixedRigidityImageName = "";
  this->GetConfiguration()->ReadParameter(
    fixedRigidityImageName, "FixedRigidityImageName", this->GetComponentLabel(), 0, -1, false);

  using RigidityImageType = typename Superclass1::RigidityImageType;
  using ChangeInfoFilterType = itk::ChangeInformationImageFilter<RigidityImageType>;
  using ChangeInfoFilterPointer = typename ChangeInfoFilterType::Pointer;
  using DirectionType = typename RigidityImageType::DirectionType;

  if (!fixedRigidityImageName.empty())
  {
    /** Use the FixedRigidityImage. */
    this->SetUseFixedRigidityImage(true);

    /** Possibly overrule the direction cosines. */
    ChangeInfoFilterPointer infoChanger = ChangeInfoFilterType::New();
    infoChanger->SetOutputDirection(DirectionType::GetIdentity());
    infoChanger->SetChangeDirection(!this->GetElastix()->GetUseDirectionCosines());

    /** Do the reading. */
    try
    {
      const auto image = itk::ReadImage<RigidityImageType>(fixedRigidityImageName);
      infoChanger->SetInput(image);
      infoChanger->Update();
    }
    catch (itk::ExceptionObject & excp)
    {
      /** Add information to the exception. */
      excp.SetLocation("MattesMutualInformationWithRigidityPenalty - BeforeRegistration()");
      std::string err_str = excp.GetDescription();
      err_str += "\nError occurred while reading the fixed rigidity image.\n";
      excp.SetDescription(err_str);
      /** Pass the exception to an higher level. */
      throw;
    }

    /** Set the fixed rigidity image into the superclass. */
    this->SetFixedRigidityImage(infoChanger->GetOutput());
  }
  else
  {
    this->SetUseFixedRigidityImage(false);
  }

  /** Read the moving rigidity image if desired. */
  std::string movingRigidityImageName = "";
  this->GetConfiguration()->ReadParameter(
    movingRigidityImageName, "MovingRigidityImageName", this->GetComponentLabel(), 0, -1, false);

  if (!movingRigidityImageName.empty())
  {
    /** Use the movingRigidityImage. */
    this->SetUseMovingRigidityImage(true);

    /** Possibly overrule the direction cosines. */
    ChangeInfoFilterPointer infoChanger = ChangeInfoFilterType::New();
    infoChanger->SetOutputDirection(DirectionType::GetIdentity());
    infoChanger->SetChangeDirection(!this->GetElastix()->GetUseDirectionCosines());

    /** Do the reading. */
    try
    {
      const auto image = itk::ReadImage<RigidityImageType>(movingRigidityImageName);
      infoChanger->SetInput(image);
      infoChanger->Update();
    }
    catch (itk::ExceptionObject & excp)
    {
      /** Add information to the exception. */
      excp.SetLocation("MattesMutualInformationWithRigidityPenalty - BeforeRegistration()");
      std::string err_str = excp.GetDescription();
      err_str += "\nError occurred while reading the moving rigidity image.\n";
      excp.SetDescription(err_str);
      /** Pass the exception to an higher level. */
      throw;
    }

    /** Set the moving rigidity image into the superclass. */
    this->SetMovingRigidityImage(infoChanger->GetOutput());
  }
  else
  {
    this->SetUseMovingRigidityImage(false);
  }

  /** Important check: at least one rigidity image must be given. */
  if (fixedRigidityImageName.empty() && movingRigidityImageName.empty())
  {
    xl::xout["warning"] << "WARNING: FixedRigidityImageName and MovingRigidityImage are both not supplied.\n"
                        << "  The rigidity penalty term is evaluated on entire input transform domain." << std::endl;
  }

  /** Add target cells to IterationInfo. */
  this->AddTargetCellToIterationInfo("Metric-LC");
  this->AddTargetCellToIterationInfo("Metric-OC");
  this->AddTargetCellToIterationInfo("Metric-PC");
  this->AddTargetCellToIterationInfo("||Gradient-LC||");
  this->AddTargetCellToIterationInfo("||Gradient-OC||");
  this->AddTargetCellToIterationInfo("||Gradient-PC||");

  /** Format the metric as floats. */
  this->GetIterationInfoAt("Metric-LC") << std::showpoint << std::fixed << std::setprecision(10);
  this->GetIterationInfoAt("Metric-OC") << std::showpoint << std::fixed << std::setprecision(10);
  this->GetIterationInfoAt("Metric-PC") << std::showpoint << std::fixed << std::setprecision(10);
  this->GetIterationInfoAt("||Gradient-LC||") << std::showpoint << std::fixed << std::setprecision(10);
  this->GetIterationInfoAt("||Gradient-OC||") << std::showpoint << std::fixed << std::setprecision(10);
  this->GetIterationInfoAt("||Gradient-PC||") << std::showpoint << std::fixed << std::setprecision(10);

} // end BeforeRegistration()


/**
 * ******************* Initialize ***********************
 */

template <class TElastix>
void
TransformRigidityPenalty<TElastix>::Initialize()
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of TransformRigidityPenalty metric took: " << static_cast<long>(timer.GetMean() * 1000)
         << " ms." << std::endl;

  /** Check stuff. */
  this->CheckUseAndCalculationBooleans();

} // end Initialize()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
TransformRigidityPenalty<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level. */
  unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Get and set the dilateRigidityImages. */
  bool dilateRigidityImages = true;
  this->GetConfiguration()->ReadParameter(
    dilateRigidityImages, "DilateRigidityImages", this->GetComponentLabel(), level, 0);
  this->SetDilateRigidityImages(dilateRigidityImages);

  /** Get and set the dilationRadiusMultiplier. */
  double dilationRadiusMultiplier = 1.0;
  this->GetConfiguration()->ReadParameter(
    dilationRadiusMultiplier, "DilationRadiusMultiplier", this->GetComponentLabel(), level, 0);
  this->SetDilationRadiusMultiplier(dilationRadiusMultiplier);

  /** Get and set the usage of the linearity condition part. */
  bool useLinearityCondition = true;
  this->GetConfiguration()->ReadParameter(
    useLinearityCondition, "UseLinearityCondition", this->GetComponentLabel(), level, 0);
  this->SetUseLinearityCondition(useLinearityCondition);

  /** Get and set the usage of the orthonormality condition part. */
  bool useOrthonormalityCondition = true;
  this->GetConfiguration()->ReadParameter(
    useOrthonormalityCondition, "UseOrthonormalityCondition", this->GetComponentLabel(), level, 0);
  this->SetUseOrthonormalityCondition(useOrthonormalityCondition);

  /** Set the usage of the properness condition part. */
  bool usePropernessCondition = true;
  this->GetConfiguration()->ReadParameter(
    usePropernessCondition, "UsePropernessCondition", this->GetComponentLabel(), level, 0);
  this->SetUsePropernessCondition(usePropernessCondition);

  /** Set the calculation of the linearity condition part. */
  bool calculateLinearityCondition = true;
  this->GetConfiguration()->ReadParameter(
    calculateLinearityCondition, "CalculateLinearityCondition", this->GetComponentLabel(), level, 0);
  this->SetCalculateLinearityCondition(calculateLinearityCondition);

  /** Set the calculation of the orthonormality condition part. */
  bool calculateOrthonormalityCondition = true;
  this->GetConfiguration()->ReadParameter(
    calculateOrthonormalityCondition, "CalculateOrthonormalityCondition", this->GetComponentLabel(), level, 0);
  this->SetCalculateOrthonormalityCondition(calculateOrthonormalityCondition);

  /** Set the calculation of the properness condition part. */
  bool calculatePropernessCondition = true;
  this->GetConfiguration()->ReadParameter(
    calculatePropernessCondition, "CalculatePropernessCondition", this->GetComponentLabel(), level, 0);
  this->SetCalculatePropernessCondition(calculatePropernessCondition);

  /** Set the LinearityConditionWeight of this level. */
  double linearityConditionWeight = 1.0;
  this->m_Configuration->ReadParameter(
    linearityConditionWeight, "LinearityConditionWeight", this->GetComponentLabel(), level, 0);
  this->SetLinearityConditionWeight(linearityConditionWeight);

  /** Set the orthonormalityConditionWeight of this level. */
  double orthonormalityConditionWeight = 1.0;
  this->m_Configuration->ReadParameter(
    orthonormalityConditionWeight, "OrthonormalityConditionWeight", this->GetComponentLabel(), level, 0);
  this->SetOrthonormalityConditionWeight(orthonormalityConditionWeight);

  /** Set the propernessConditionWeight of this level. */
  double propernessConditionWeight = 1.0;
  this->m_Configuration->ReadParameter(
    propernessConditionWeight, "PropernessConditionWeight", this->GetComponentLabel(), level, 0);
  this->SetPropernessConditionWeight(propernessConditionWeight);

} // end BeforeEachResolution()


/**
 * ***************AfterEachIteration ****************************
 */

template <class TElastix>
void
TransformRigidityPenalty<TElastix>::AfterEachIteration()
{
  /** Print some information. */
  this->GetIterationInfoAt("Metric-LC") << this->GetLinearityConditionValue();
  this->GetIterationInfoAt("Metric-OC") << this->GetOrthonormalityConditionValue();
  this->GetIterationInfoAt("Metric-PC") << this->GetPropernessConditionValue();

  this->GetIterationInfoAt("||Gradient-LC||") << this->GetLinearityConditionGradientMagnitude();
  this->GetIterationInfoAt("||Gradient-OC||") << this->GetOrthonormalityConditionGradientMagnitude();
  this->GetIterationInfoAt("||Gradient-PC||") << this->GetPropernessConditionGradientMagnitude();

} // end AfterEachIteration()


} // end namespace elastix

#endif // end #ifndef elxTransformRigidityPenaltyTerm_hxx
