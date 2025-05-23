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

#ifndef elxSimilarityTransform_hxx
#define elxSimilarityTransform_hxx

#include "elxSimilarityTransform.h"
#include "itkContinuousIndex.h"
#include <itkDeref.h>

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template <typename TElastix>
SimilarityTransformElastix<TElastix>::SimilarityTransformElastix()
{
  this->SetCurrentTransform(this->m_SimilarityTransform);

} // end Constructor()


/**
 * ******************* BeforeRegistration ***********************
 */

template <typename TElastix>
void
SimilarityTransformElastix<TElastix>::BeforeRegistration()
{
  /** Task 1 - Set center of rotation and initial translation. */
  this->InitializeTransform();

  /** Task 2 - Set the scales. */
  this->SetScales();

} // end BeforeRegistration()


/**
 * ************************* ReadFromFile ************************
 */

template <typename TElastix>
void
SimilarityTransformElastix<TElastix>::ReadFromFile()
{
  if (!this->HasITKTransformParameters())
  {
    /** Variables. */
    InputPointType centerOfRotationPoint{};

    /** Try to read the CenterOfRotationPoint from the transform parameter file
     */
    const bool pointRead = this->ReadCenterOfRotationPoint(centerOfRotationPoint);

    if (!pointRead)
    {
      log::error("ERROR: No center of rotation is specified in the transform parameter file.");
      itkExceptionMacro("Transform parameter file is corrupt.");
    }

    /** Set the center in this Transform. */
    this->m_SimilarityTransform->SetCenter(centerOfRotationPoint);
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
SimilarityTransformElastix<TElastix>::CreateDerivedTransformParameterMap() const -> ParameterMapType
{
  return { { "CenterOfRotationPoint", Conversion::ToVectorOfStrings(m_SimilarityTransform->GetCenter()) } };

} // end CreateDerivedTransformParameterMap()


/**
 * ************************* InitializeTransform *********************
 */

template <typename TElastix>
void
SimilarityTransformElastix<TElastix>::InitializeTransform()
{
  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

  /** Set the parameters to mimic the identity transform. */
  this->m_SimilarityTransform->SetIdentity();

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

  /** Check if CenterOfRotation has index-values within image.*/
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
    transformInitializer->SetTransform(this->m_SimilarityTransform);

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
    this->m_SimilarityTransform->SetTranslation(noTranslation);
  }

  /** Set the center of rotation if it was entered by the user. */
  if (centerGiven)
  {
    if (centerGivenAsIndex)
    {
      /** Convert from index-value to physical-point-value.*/
      this->m_Registration->GetAsITKBaseType()->GetFixedImage()->TransformIndexToPhysicalPoint(centerOfRotationIndex,
                                                                                               centerOfRotationPoint);
    }
    this->m_SimilarityTransform->SetCenter(centerOfRotationPoint);
  }

  /** Apply the initial transform to the center of rotation, if
   * composition is used to combine the initial transform with the
   * the current (euler) transform.
   */
  if (const auto * const initialTransform = this->Superclass1::GetInitialTransform();
      initialTransform != nullptr && this->GetUseComposition())
  {
    InputPointType transformedCenterOfRotationPoint =
      initialTransform->TransformPoint(this->m_SimilarityTransform->GetCenter());
    this->m_SimilarityTransform->SetCenter(transformedCenterOfRotationPoint);
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
SimilarityTransformElastix<TElastix>::SetScales()
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
    /** If the dimension is 2, then the first parameter represents the
     * isotropic scaling, the second the rotation, and the third and
     * fourth the translation.
     * If the dimension is 3, then the first three represent rotations,
     * the second three translations, and the last parameter the isotropic
     * scaling.
     */

    /** Create the scales and set to default values. */
    if constexpr (SpaceDimension == 2)
    {
      newscales[0] = 10000.0;
      newscales[1] = 100000.0;
    }
    else if constexpr (SpaceDimension == 3)
    {
      newscales[6] = 10000.0;
      for (unsigned int i = 0; i < 3; ++i)
      {
        newscales[i] = 100000.0;
      }
    }

    /** Get the scales from the parameter file. */
    for (unsigned int i = 0; i < numberOfParameters; ++i)
    {
      this->GetConfiguration()->ReadParameter(newscales[i], "Scales", this->GetComponentLabel(), i, -1);
    }
  } // end else: no automatic parameter estimation

  log::info(std::ostringstream{} << "Scales for transform parameters are: " << newscales);

  /** Set the scales into the optimizer. */
  this->m_Registration->GetAsITKBaseType()->GetModifiableOptimizer()->SetScales(newscales);

} // end SetScales()


/**
 * ******************** ReadCenterOfRotationPoint *********************
 */

template <typename TElastix>
bool
SimilarityTransformElastix<TElastix>::ReadCenterOfRotationPoint(InputPointType & rotationPoint) const
{
  /** Try to read CenterOfRotationPoint from the transform parameter
   * file, which is the rotationPoint, expressed in world coordinates.
   */
  InputPointType centerOfRotationPoint{};
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    /** Returns zero when parameter was in the parameter file. */
    bool found = this->m_Configuration->ReadParameter(centerOfRotationPoint[i], "CenterOfRotationPoint", i, false);
    if (!found)
    {
      return false;
    }
  }

  /** copy the temporary variable into the output of this function,
   * if everything went ok.
   */
  rotationPoint = centerOfRotationPoint;

  /** Successfully read centerOfRotation as Point. */
  return true;

} // end ReadCenterOfRotationPoint()


} // end namespace elastix

#endif // end #ifndef elxSimilarityTransform_hxx
