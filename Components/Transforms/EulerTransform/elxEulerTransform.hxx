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

#ifndef elxEulerTransform_hxx
#define elxEulerTransform_hxx

#include "elxEulerTransform.h"
#include "itkImageGridSampler.h"
#include "itkContinuousIndex.h"
#include <itkDeref.h>

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template <typename TElastix>
EulerTransformElastix<TElastix>::EulerTransformElastix()
{
  this->SetCurrentTransform(this->m_EulerTransform);

} // end Constructor


/**
 * ******************* BeforeRegistration ***********************
 */

template <typename TElastix>
void
EulerTransformElastix<TElastix>::BeforeRegistration()
{
  /** Set center of rotation and initial translation. */
  this->InitializeTransform();

  /** Set the scales. */
  this->SetScales();

} // end BeforeRegistration()


/**
 * ************************* ReadFromFile ************************
 */

template <typename TElastix>
void
EulerTransformElastix<TElastix>::ReadFromFile()
{
  if (!this->HasITKTransformParameters())
  {
    /** Variables. */
    InputPointType centerOfRotationPoint{};

    /** Try first to read the CenterOfRotationPoint from the
     * transform parameter file, this is the new, and preferred
     * way, since elastix 3.402.
     */
    const bool pointRead = this->ReadCenterOfRotationPoint(centerOfRotationPoint);

    if (!pointRead)
    {
      log::error("ERROR: No center of rotation is specified in the transform parameter file");
      itkExceptionMacro("Transform parameter file is corrupt.");
    }

    /** Set the center in this Transform. */
    this->m_EulerTransform->SetCenter(centerOfRotationPoint);

    /** Read the ComputeZYX. */
    if constexpr (SpaceDimension == 3)
    {
      std::string computeZYX = "false";
      this->m_Configuration->ReadParameter(computeZYX, "ComputeZYX", 0);
      if (computeZYX == "true")
      {
        this->m_EulerTransform->SetComputeZYX(true);
      }
    }
  }

  /** Call the ReadFromFile from the TransformBase.
   * BE AWARE: Only call Superclass2::ReadFromFile() after CenterOfRotationPoint
   * is set, because it is used in the SetParameters()-function of this transform.
   */
  this->Superclass2::ReadFromFile();

} // end ReadFromFile()


/**
 * ************************* CreateDerivedTransformParameterMap ************************
 */

template <typename TElastix>
auto
EulerTransformElastix<TElastix>::CreateDerivedTransformParameterMap() const -> ParameterMapType
{
  ParameterMapType parameterMap{ { "CenterOfRotationPoint",
                                   Conversion::ToVectorOfStrings(m_EulerTransform->GetCenter()) } };

  /** Write the ComputeZYX to file. */
  if constexpr (SpaceDimension == 3)
  {
    parameterMap["ComputeZYX"] = { Conversion::ToString(m_EulerTransform->GetComputeZYX()) };
  }

  return parameterMap;

} // end CreateDerivedTransformParameterMap()


/**
 * ************************* InitializeTransform *********************
 */

template <typename TElastix>
void
EulerTransformElastix<TElastix>::InitializeTransform()
{
  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

  /** Set all parameters to zero (no rotations, no translation). */
  this->m_EulerTransform->SetIdentity();

  /** Try to read CenterOfRotationIndex from parameter file,
   * which is the rotationPoint, expressed in index-values.
   */
  IndexType      centerOfRotationIndex{};
  InputPointType centerOfRotationPoint{};
  const bool     centerGivenAsIndex = [&configuration, &centerOfRotationIndex] {
    for (unsigned int i = 0; i < SpaceDimension; ++i)
    {
      if (!configuration.ReadParameter(centerOfRotationIndex[i], "CenterOfRotation", i, false))
      {
        return false;
      }
    }
    return true;
  }();
  const bool centerGivenAsPoint = ReadCenterOfRotationPoint(centerOfRotationPoint);

  /** Check if CenterOfRotation has index-values within image. */
  bool CORIndexInImage = true;
  bool CORPointInImage = true;
  if (centerGivenAsIndex)
  {
    CORIndexInImage = this->m_Registration->GetAsITKBaseType()->GetFixedImage()->GetLargestPossibleRegion().IsInside(
      centerOfRotationIndex);
  }

  if (centerGivenAsPoint)
  {
    using ContinuousIndexType = itk::ContinuousIndex<double, SpaceDimension>;
    ContinuousIndexType cindex;
    CORPointInImage =
      this->m_Registration->GetAsITKBaseType()->GetFixedImage()->TransformPhysicalPointToContinuousIndex(
        centerOfRotationPoint, cindex);
  }

  /** Give a warning if necessary. */
  if (!CORIndexInImage && centerGivenAsIndex)
  {
    log::warn("WARNING: Center of Rotation (index) is not within image boundaries!");
  }

  /** Give a warning if necessary. */
  if (!CORPointInImage && centerGivenAsPoint && !centerGivenAsIndex)
  {
    log::warn("WARNING: Center of Rotation (point) is not within image boundaries!");
  }

  /** Check if user wants automatic transform initialization; false by default.
   * If an initial transform is given, automatic transform initialization is
   * not possible.
   */
  bool automaticTransformInitialization = false;
  bool tmpBool = false;
  configuration.ReadParameter(tmpBool, "AutomaticTransformInitialization", 0);
  if (tmpBool && this->Superclass1::GetInitialTransform() == nullptr)
  {
    automaticTransformInitialization = true;
  }

  /**
   * Run the itkTransformInitializer if:
   * - No center of rotation was given, or
   * - The user asked for AutomaticTransformInitialization
   */
  bool centerGiven = centerGivenAsIndex || centerGivenAsPoint;
  if (!centerGiven || automaticTransformInitialization)
  {

    /** Use the TransformInitializer to determine a center of
     * of rotation and an initial translation.
     */
    TransformInitializerPointer transformInitializer = TransformInitializerType::New();
    transformInitializer->SetFixedImage(this->m_Registration->GetAsITKBaseType()->GetFixedImage());
    transformInitializer->SetMovingImage(this->m_Registration->GetAsITKBaseType()->GetMovingImage());
    transformInitializer->SetTransform(this->m_EulerTransform);

    /** Select the method of initialization. Default: "GeometricalCenter". */
    transformInitializer->GeometryOn();
    std::string method = "GeometricalCenter";
    configuration.ReadParameter(method, "AutomaticTransformInitializationMethod", 0);
    if (method == "CenterOfGravity")
    {
      transformInitializer->MomentsOn();
    }

    transformInitializer->InitializeTransform();
  }

  /** Set the translation to zero, if no AutomaticTransformInitialization
   * was desired.
   */
  if (!automaticTransformInitialization)
  {
    OutputVectorType noTranslation{};
    this->m_EulerTransform->SetTranslation(noTranslation);
  }

  /** Set the center of rotation if it was entered by the user. */
  if (centerGiven)
  {
    if (centerGivenAsIndex)
    {
      /** Convert from index-value to physical-point-value. */
      this->m_Registration->GetAsITKBaseType()->GetFixedImage()->TransformIndexToPhysicalPoint(centerOfRotationIndex,
                                                                                               centerOfRotationPoint);
    }
    this->m_EulerTransform->SetCenter(centerOfRotationPoint);
  }

  /** Apply the initial transform to the center of rotation, if
   * composition is used to combine the initial transform with the
   * the current (euler) transform.
   */
  if (const auto * const initialTransform = this->Superclass1::GetInitialTransform();
      initialTransform != nullptr && this->GetUseComposition())
  {
    InputPointType transformedCenterOfRotationPoint =
      initialTransform->TransformPoint(this->m_EulerTransform->GetCenter());
    this->m_EulerTransform->SetCenter(transformedCenterOfRotationPoint);
  }

  /** Set the initial parameters in this->m_Registration. */
  this->m_Registration->GetAsITKBaseType()->SetInitialTransformParameters(this->GetParameters());

  /** Give feedback. */
  // \todo: should perhaps also print fixed parameters
  log::info(std::ostringstream{} << "Transform parameters are initialized as: " << this->GetParameters());

} // end InitializeTransform()


/**
 * ************************* SetScales *********************
 */

template <typename TElastix>
void
EulerTransformElastix<TElastix>::SetScales()
{
  /** Create the new scales. */
  const NumberOfParametersType numberOfParameters = this->GetNumberOfParameters();
  ScalesType                   newscales(numberOfParameters, 1.0);

  /** Check if automatic scales estimation is desired. */
  bool automaticScalesEstimation = false;
  this->m_Configuration->ReadParameter(automaticScalesEstimation, "AutomaticScalesEstimation", 0);

  if (automaticScalesEstimation)
  {
    log::info("Scales are estimated automatically.");
    this->AutomaticScalesEstimation(newscales);
  }
  else
  {

    /** Here is an heuristic rule for estimating good values for
     * the rotation/translation scales.
     *
     * 1) Estimate the bounding box of your points (in physical units).
     * 2) Take the 3D Diagonal of that bounding box
     * 3) Multiply that by 10.0.
     * 4) use 1.0 /[ value from (3) ] as the translation scaling value.
     * 5) use 1.0 as the rotation scaling value.
     *
     * With this operation you bring the translation units
     * to the range of rotations (e.g. around -1 to 1).
     * After that, all your registration parameters are
     * in the relaxed range of -1:1. At that point you
     * can start setting your optimizer with step lengths
     * in the ranges of 0.001 if you are conservative, or
     * in the range of 0.1 if you want to live dangerously.
     * (0.1 radians is about 5.7 degrees).
     *
     * This heuristic rule is based on the naive assumption
     * that your registration may require translations as
     * large as 1/10 of the diagonal of the bounding box.
     */

    const double defaultScalingvalue = 100000.0;

    /** If the Dimension is 3, the first 3 parameters represent rotations.
     * If the Dimension is 2, only the first parameter represent a rotation.
     */
    unsigned int RotationPart = 3;
    if constexpr (SpaceDimension == 2)
    {
      RotationPart = 1;
    }

    /** this->m_Configuration->ReadParameter() returns 0 if there is a value given
     * in the parameter-file, and returns 1 if there is no value given in the
     * parameter-file.
     * Check which option is used:
     * - Nothing given in the parameter-file: rotations are scaled by the default
     *   value 100000.0
     * - Only one scale given in the parameter-file: rotations are scaled by this
     *   value.
     * - All scales are given in the parameter-file: each parameter is assigned its
     *   own scale.
     */

    std::size_t count = this->m_Configuration->CountNumberOfParameterEntries("Scales");

    /** Check which of the above options is used. */
    if (count == 0)
    {
      /** In this case the first option is used. */
      for (unsigned int i = 0; i < RotationPart; ++i)
      {
        newscales[i] = defaultScalingvalue;
      }
    }
    else if (count == 1)
    {
      /** In this case the second option is used. */
      double scale = defaultScalingvalue;
      this->m_Configuration->ReadParameter(scale, "Scales", 0);
      for (unsigned int i = 0; i < RotationPart; ++i)
      {
        newscales[i] = scale;
      }
    }
    else if (count == numberOfParameters)
    {
      /** In this case the third option is used. */
      for (unsigned int i = 0; i < numberOfParameters; ++i)
      {
        this->m_Configuration->ReadParameter(newscales[i], "Scales", i);
      }
    }
    else
    {
      /** In this case an error is made in the parameter-file.
       * An error is thrown, because using erroneous scales in the optimizer
       * can give unpredictable results.
       */
      itkExceptionMacro("ERROR: The Scales-option in the parameter-file has not been set properly.");
    }

  } // end else: no automaticScalesEstimation

  log::info(std::ostringstream{} << "Scales for transform parameters are: " << newscales);

  /** Set the scales into the optimizer. */
  this->m_Registration->GetAsITKBaseType()->GetModifiableOptimizer()->SetScales(newscales);

} // end SetScales()


/**
 * ******************** ReadCenterOfRotationPoint *********************
 */

template <typename TElastix>
bool
EulerTransformElastix<TElastix>::ReadCenterOfRotationPoint(InputPointType & rotationPoint) const
{
  /** Try to read CenterOfRotationPoint from the transform parameter
   * file, which is the rotationPoint, expressed in world coordinates.
   */
  InputPointType centerOfRotationPoint{};
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    /** Returns zero when parameter was in the parameter file */
    bool found = this->m_Configuration->ReadParameter(centerOfRotationPoint[i], "CenterOfRotationPoint", i, false);
    if (!found)
    {
      return false;
    }
  }

  /** Copy the temporary variable into the output of this function,
   * if everything went ok.
   */
  rotationPoint = centerOfRotationPoint;

  /** Successfully read centerOfRotation as Point. */
  return true;

} // end ReadCenterOfRotationPoint()


} // end namespace elastix

#endif // end #ifndef elxEulerTransform_hxx
