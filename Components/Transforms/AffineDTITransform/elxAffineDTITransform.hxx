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
#ifndef elxAffineDTITransform_hxx
#define elxAffineDTITransform_hxx

#include "elxAffineDTITransform.h"
#include "itkContinuousIndex.h"

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template <class TElastix>
AffineDTITransformElastix<TElastix>::AffineDTITransformElastix()
{
  this->SetCurrentTransform(this->m_AffineDTITransform);

} // end Constructor


/**
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
void
AffineDTITransformElastix<TElastix>::BeforeRegistration()
{
  if (SpaceDimension != 2 && SpaceDimension != 3)
  {
    itkExceptionMacro(<< "AffineDTI transform only works for 2D or 3D images!");
  }

  /** Set center of rotation and initial translation. */
  this->InitializeTransform();

  /** Set the scales. */
  this->SetScales();

} // end BeforeRegistration()


/**
 * ************************* ReadFromFile ************************
 */

template <class TElastix>
void
AffineDTITransformElastix<TElastix>::ReadFromFile()
{
  /** Variables. */
  InputPointType centerOfRotationPoint;
  centerOfRotationPoint.Fill(0.0);

  /** Try first to read the CenterOfRotationPoint from the
   * transform parameter file, this is the new, and preferred
   * way, since elastix 3.402.
   */
  const bool pointRead = this->ReadCenterOfRotationPoint(centerOfRotationPoint);

  if (!pointRead)
  {
    xl::xout["error"] << "ERROR: No center of rotation is specified in the transform parameter file" << std::endl;
    itkExceptionMacro(<< "Transform parameter file is corrupt.")
  }

  /** Set the center in this Transform. */
  this->m_AffineDTITransform->SetCenter(centerOfRotationPoint);

  /** Call the ReadFromFile from the TransformBase.
   * BE AWARE: Only call Superclass2::ReadFromFile() after CenterOfRotation
   * is set, because it is used in the SetParameters()-function of this transform.
   */
  this->Superclass2::ReadFromFile();

} // end ReadFromFile()


/**
 * ************************* CreateDerivedTransformParametersMap ************************
 */

template <class TElastix>
auto
AffineDTITransformElastix<TElastix>::CreateDerivedTransformParametersMap() const -> ParameterMapType
{
  const auto & itkTransform = *m_AffineDTITransform;

  return { { "CenterOfRotationPoint", Conversion::ToVectorOfStrings(itkTransform.GetCenter()) },
           { "MatrixTranslation",
             Conversion::ConcatenateVectors(Conversion::ToVectorOfStrings(itkTransform.GetMatrix()),
                                            Conversion::ToVectorOfStrings(itkTransform.GetTranslation())) } };

} // end CreateDerivedTransformParametersMap()


/**
 * ************************* InitializeTransform *********************
 */

template <class TElastix>
void
AffineDTITransformElastix<TElastix>::InitializeTransform()
{
  /** Set all parameters to zero (no rotations, no translation). */
  this->m_AffineDTITransform->SetIdentity();

  /** Try to read CenterOfRotationIndex from parameter file,
   * which is the rotationPoint, expressed in index-values.
   */
  IndexType      centerOfRotationIndex;
  InputPointType centerOfRotationPoint;
  bool           centerGivenAsIndex = true;
  bool           centerGivenAsPoint = true;
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    /** Initialize. */
    centerOfRotationIndex[i] = 0;
    centerOfRotationPoint[i] = 0.0;

    /** Check COR index: Returns zero when parameter was in the parameter file. */
    bool foundI = this->m_Configuration->ReadParameter(centerOfRotationIndex[i], "CenterOfRotation", i, false);
    if (!foundI)
    {
      centerGivenAsIndex = false;
    }

    /** Check COR point: Returns zero when parameter was in the parameter file. */
    bool foundP = this->m_Configuration->ReadParameter(centerOfRotationPoint[i], "CenterOfRotationPoint", i, false);
    if (!foundP)
    {
      centerGivenAsPoint = false;
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
    xl::xout["warning"] << "WARNING: Center of Rotation (index) is not within image boundaries!" << std::endl;
  }

  /** Give a warning if necessary. */
  if (!CORPointInImage && centerGivenAsPoint && !centerGivenAsIndex)
  {
    xl::xout["warning"] << "WARNING: Center of Rotation (point) is not within image boundaries!" << std::endl;
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
    transformInitializer->SetTransform(this->m_AffineDTITransform);

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
    OutputVectorType noTranslation;
    noTranslation.Fill(0.0);
    this->m_AffineDTITransform->SetTranslation(noTranslation);
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
    this->m_AffineDTITransform->SetCenter(centerOfRotationPoint);
  }

  /** Apply the initial transform to the center of rotation, if
   * composition is used to combine the initial transform with the
   * the current (euler) transform.
   */
  if (this->GetUseComposition() && this->Superclass1::GetInitialTransform() != nullptr)
  {
    InputPointType transformedCenterOfRotationPoint =
      this->Superclass1::GetInitialTransform()->TransformPoint(this->m_AffineDTITransform->GetCenter());
    this->m_AffineDTITransform->SetCenter(transformedCenterOfRotationPoint);
  }

  /** Set the initial parameters in this->m_Registration. */
  this->m_Registration->GetAsITKBaseType()->SetInitialTransformParameters(this->GetParameters());

  /** Give feedback. */
  // \todo: should perhaps also print fixed parameters
  elxout << "Transform parameters are initialized as: " << this->GetParameters() << std::endl;

} // end InitializeTransform()


/**
 * ************************* SetScales *********************
 */

template <class TElastix>
void
AffineDTITransformElastix<TElastix>::SetScales()
{
  /** Create the new scales. */
  const NumberOfParametersType numberOfParameters = this->GetNumberOfParameters();
  ScalesType                   newscales(numberOfParameters);
  newscales.Fill(1.0);

  /** Always estimate scales automatically */
  elxout << "Scales are estimated automatically." << std::endl;
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
    itkExceptionMacro(<< "ERROR: The Scales-option in the parameter-file has not been set properly.");
  }

  elxout << "Scales for transform parameters are: " << newscales << std::endl;

  /** Set the scales into the optimizer. */
  this->m_Registration->GetAsITKBaseType()->GetModifiableOptimizer()->SetScales(newscales);

} // end SetScales()


/**
 * ******************** ReadCenterOfRotationPoint *********************
 */

template <class TElastix>
bool
AffineDTITransformElastix<TElastix>::ReadCenterOfRotationPoint(InputPointType & rotationPoint) const
{
  /** Try to read CenterOfRotationPoint from the transform parameter
   * file, which is the rotationPoint, expressed in world coordinates.
   */
  InputPointType centerOfRotationPoint;
  bool           centerGivenAsPoint = true;
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    centerOfRotationPoint[i] = 0;

    /** Returns zero when parameter was in the parameter file */
    bool found = this->m_Configuration->ReadParameter(centerOfRotationPoint[i], "CenterOfRotationPoint", i, false);
    if (!found)
    {
      centerGivenAsPoint = false;
    }
  }

  if (!centerGivenAsPoint)
  {
    return false;
  }

  /** Copy the temporary variable into the output of this function,
   * if everything went ok.
   */
  rotationPoint = centerOfRotationPoint;

  /** Successfully read centerOfRotation as Point. */
  return true;

} // end ReadCenterOfRotationPoint()


} // end namespace elastix

#endif // end #ifndef elxAffineDTITransform_hxx
