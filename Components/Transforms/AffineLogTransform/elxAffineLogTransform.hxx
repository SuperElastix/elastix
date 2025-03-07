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
#ifndef elxAffineLogTransform_hxx
#define elxAffineLogTransform_hxx

#include "elxAffineLogTransform.h"
#include "itkContinuousIndex.h"

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template <typename TElastix>
AffineLogTransformElastix<TElastix>::AffineLogTransformElastix()
{
  log::info("Constructor");
  this->SetCurrentTransform(this->m_AffineLogTransform);

} // end Constructor


/**
 * ******************* BeforeRegistration ***********************
 */

template <typename TElastix>
void
AffineLogTransformElastix<TElastix>::BeforeRegistration()
{
  log::info("BeforeRegistration");
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
AffineLogTransformElastix<TElastix>::ReadFromFile()
{
  log::info("ReadFromFile");
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
  this->m_AffineLogTransform->SetCenter(centerOfRotationPoint);

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
AffineLogTransformElastix<TElastix>::CreateDerivedTransformParameterMap() const -> ParameterMapType
{
  const auto & itkTransform = *m_AffineLogTransform;

  return { { "CenterOfRotationPoint", Conversion::ToVectorOfStrings(itkTransform.GetCenter()) },
           { "MatrixTranslation",
             Conversion::ConcatenateVectors(Conversion::ToVectorOfStrings(itkTransform.GetMatrix()),
                                            Conversion::ToVectorOfStrings(itkTransform.GetTranslation())) } };

} // end CreateDerivedTransformParameterMap()


/**
 * ************************* InitializeTransform *********************
 */

template <typename TElastix>
void
AffineLogTransformElastix<TElastix>::InitializeTransform()
{
  log::info("InitializeTransform");
  /** Set all parameters to zero (no rotations, no translation). */
  this->m_AffineLogTransform->SetIdentity();

  /** Try to read CenterOfRotationIndex from parameter file,
   * which is the rotationPoint, expressed in index-values.
   */
  IndexType      centerOfRotationIndex{};
  InputPointType centerOfRotationPoint{};
  bool           centerGivenAsIndex = true;
  const bool     centerGivenAsPoint = ReadCenterOfRotationPoint(centerOfRotationPoint);
  // SizeType fixedImageSize = this->m_Registration->GetAsITKBaseType()
  //  ->GetFixedImage()->GetLargestPossibleRegion().GetSize();
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    /** Check COR index: Returns zero when parameter was in the parameter file. */
    bool foundI = this->m_Configuration->ReadParameter(centerOfRotationIndex[i], "CenterOfRotation", i, false);
    if (!foundI)
    {
      centerGivenAsIndex = false;
    }
  } // end loop over SpaceDimension

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
  this->m_Configuration->ReadParameter(tmpBool, "AutomaticTransformInitialization", 0);
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
    transformInitializer->SetTransform(this->m_AffineLogTransform);

    /** Select the method of initialization. Default: "GeometricalCenter". */
    transformInitializer->GeometryOn();
    std::string method = "GeometricalCenter";
    this->m_Configuration->ReadParameter(method, "AutomaticTransformInitializationMethod", 0);
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
    this->m_AffineLogTransform->SetTranslation(noTranslation);
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
    log::info(std::ostringstream{} << "cor: " << centerOfRotationPoint);

    this->m_AffineLogTransform->SetCenter(centerOfRotationPoint);
  }

  /** Apply the initial transform to the center of rotation, if
   * composition is used to combine the initial transform with the
   * the current (euler) transform.
   */
  if (const auto * const initialTransform = this->Superclass1::GetInitialTransform();
      initialTransform != nullptr && this->GetUseComposition())
  {
    InputPointType transformedCenterOfRotationPoint =
      initialTransform->TransformPoint(this->m_AffineLogTransform->GetCenter());
    this->m_AffineLogTransform->SetCenter(transformedCenterOfRotationPoint);
  }

  /** Set the initial parameters in this->m_Registration. */
  this->m_Registration->GetAsITKBaseType()->SetInitialTransformParameters(this->GetParameters());

} // end InitializeTransform()


/**
 * ************************* SetScales *********************
 */

template <typename TElastix>
void
AffineLogTransformElastix<TElastix>::SetScales()
{
  log::info("SetScales");
  /** Create the new scales. */
  const NumberOfParametersType numberOfParameters = this->GetNumberOfParameters();
  ScalesType                   newscales(numberOfParameters, 1.0);

  /** Always estimate scales automatically */
  log::info("Scales are estimated automatically.");
  this->AutomaticScalesEstimation(newscales);

  std::size_t count = this->m_Configuration->CountNumberOfParameterEntries("Scales");

  if (count == numberOfParameters)
  {
    /** Overrule the automatically estimated scales with the user-specified
     * scales. Values <= 0 are not used; the default is kept then. */
    for (unsigned int i = 0; i < numberOfParameters; ++i)
    {
      double scale_i = -1.0;
      this->m_Configuration->ReadParameter(scale_i, "Scales", i);
      if (scale_i > 0)
      {
        newscales[i] = scale_i;
      }
    }
  }
  else if (count != 0)
  {
    /** In this case an error is made in the parameter-file.
     * An error is thrown, because using erroneous scales in the optimizer
     * can give unpredictable results.
     */
    itkExceptionMacro("ERROR: The Scales-option in the parameter-file has not been set properly.");
  }

  log::info(std::ostringstream{} << "Scales for transform parameters are: " << newscales);

  /** Set the scales into the optimizer. */
  this->m_Registration->GetAsITKBaseType()->GetModifiableOptimizer()->SetScales(newscales);

} // end SetScales()


/**
 * ******************** ReadCenterOfRotationPoint *********************
 */

template <typename TElastix>
bool
AffineLogTransformElastix<TElastix>::ReadCenterOfRotationPoint(InputPointType & rotationPoint) const
{
  log::info("ReadCenterOfRotationPoint");
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

#endif // ELXAFFINELOGTRANSFORM_HXX
