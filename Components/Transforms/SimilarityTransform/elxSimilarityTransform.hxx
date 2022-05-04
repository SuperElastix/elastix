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

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template <class TElastix>
SimilarityTransformElastix<TElastix>::SimilarityTransformElastix()
{
  this->SetCurrentTransform(this->m_SimilarityTransform);

} // end Constructor()


/**
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
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

template <class TElastix>
void
SimilarityTransformElastix<TElastix>::ReadFromFile()
{
  if (!this->HasITKTransformParameters())
  {
    /** Variables. */
    InputPointType centerOfRotationPoint;
    centerOfRotationPoint.Fill(0.0);
    bool indexRead = false;

    /** Try first to read the CenterOfRotationPoint from the
     * transform parameter file, this is the new, and preferred
     * way, since elastix 3.402.
     */
    const bool pointRead = this->ReadCenterOfRotationPoint(centerOfRotationPoint);

    /** If this did not succeed, probably a transform parameter file
     * is trying to be read that was generated using an older elastix
     * version. Try to read it as an index, and convert to point.
     */
    if (!pointRead)
    {
      indexRead = this->ReadCenterOfRotationIndex(centerOfRotationPoint);
    }

    if (!pointRead && !indexRead)
    {
      xl::xout["error"] << "ERROR: No center of rotation is specified in the transform parameter file." << std::endl;
      itkExceptionMacro(<< "Transform parameter file is corrupt.")
    }

    /** Set the center in this Transform. */
    this->m_SimilarityTransform->SetCenter(centerOfRotationPoint);
  }

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
SimilarityTransformElastix<TElastix>::CreateDerivedTransformParametersMap() const -> ParameterMapType
{
  return { { "CenterOfRotationPoint", Conversion::ToVectorOfStrings(m_SimilarityTransform->GetCenter()) } };

} // end CreateDerivedTransformParametersMap()


/**
 * ************************* InitializeTransform *********************
 */

template <class TElastix>
void
SimilarityTransformElastix<TElastix>::InitializeTransform()
{
  /** Set the parameters to mimic the identity transform. */
  this->m_SimilarityTransform->SetIdentity();

  /** Try to read CenterOfRotationIndex from parameter file,
   * which is the rotationPoint, expressed in index-values.
   */
  IndexType      centerOfRotationIndex;
  InputPointType centerOfRotationPoint;
  bool           centerGivenAsIndex = true;
  bool           centerGivenAsPoint = true;
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    /** Initilialize. */
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
    transformInitializer->SetTransform(this->m_SimilarityTransform);

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
  if (this->GetUseComposition() && this->Superclass1::GetInitialTransform() != nullptr)
  {
    InputPointType transformedCenterOfRotationPoint =
      this->Superclass1::GetInitialTransform()->TransformPoint(this->m_SimilarityTransform->GetCenter());
    this->m_SimilarityTransform->SetCenter(transformedCenterOfRotationPoint);
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
SimilarityTransformElastix<TElastix>::SetScales()
{
  /** Create the new scales. */
  const NumberOfParametersType numberOfParameters = this->GetNumberOfParameters();
  ScalesType                   newscales(numberOfParameters);
  newscales.Fill(1.0);

  /** Check if automatic scales estimation is desired. */
  bool automaticScalesEstimation = false;
  this->m_Configuration->ReadParameter(automaticScalesEstimation, "AutomaticScalesEstimation", 0);

  if (automaticScalesEstimation)
  {
    elxout << "Scales are estimated automatically." << std::endl;
    this->AutomaticScalesEstimation(newscales);
  }
  else
  {
    /** If the dimension is 2, then the first parameter represents the
     * isotropic scaling, the the second the rotation, and the third and
     * fourth the translation.
     * If the dimension is 3, then the first three represent rotations,
     * the second three translations, and the last parameter the isotropic
     * scaling.
     */

    /** Create the scales and set to default values. */
    if (SpaceDimension == 2)
    {
      newscales[0] = 10000.0;
      newscales[1] = 100000.0;
    }
    else if (SpaceDimension == 3)
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

  elxout << "Scales for transform parameters are: " << newscales << std::endl;

  /** Set the scales into the optimizer. */
  this->m_Registration->GetAsITKBaseType()->GetModifiableOptimizer()->SetScales(newscales);

} // end SetScales()


/**
 * ******************** ReadCenterOfRotationIndex *********************
 */

template <class TElastix>
bool
SimilarityTransformElastix<TElastix>::ReadCenterOfRotationIndex(InputPointType & rotationPoint) const
{
  /** Try to read CenterOfRotationIndex from the transform parameter
   * file, which is the rotationPoint, expressed in index-values.
   */
  IndexType centerOfRotationIndex;
  bool      centerGivenAsIndex = true;
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    centerOfRotationIndex[i] = 0;

    /** Returns zero when parameter was in the parameter file. */
    bool found = this->m_Configuration->ReadParameter(centerOfRotationIndex[i], "CenterOfRotation", i, false);
    if (!found)
    {
      centerGivenAsIndex = false;
    }
  }

  if (!centerGivenAsIndex)
  {
    return false;
  }

  /** Get spacing, origin and size of the fixed image.
   * We put this in a dummy image, so that we can correctly
   * calculate the center of rotation in world coordinates.
   */
  SpacingType spacing;
  IndexType   index;
  PointType   origin;
  SizeType    size;
  auto        direction = DirectionType::GetIdentity();
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    /** Read size from the parameter file. Zero by default, which is illegal. */
    size[i] = 0;
    this->m_Configuration->ReadParameter(size[i], "Size", i);

    /** Default index. Read index from the parameter file. */
    index[i] = 0;
    this->m_Configuration->ReadParameter(index[i], "Index", i);

    /** Default spacing. Read spacing from the parameter file. */
    spacing[i] = 1.0;
    this->m_Configuration->ReadParameter(spacing[i], "Spacing", i);

    /** Default origin. Read origin from the parameter file. */
    origin[i] = 0.0;
    this->m_Configuration->ReadParameter(origin[i], "Origin", i);

    /** Read direction cosines. Default identity */
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      this->m_Configuration->ReadParameter(direction(j, i), "Direction", i * SpaceDimension + j);
    }
  }

  /** Check for image size. */
  bool illegalSize = false;
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    if (size[i] == 0)
    {
      illegalSize = true;
    }
  }

  if (illegalSize)
  {
    xl::xout["error"] << "ERROR: One or more image sizes are 0!" << std::endl;
    return false;
  }

  /** Make a temporary image with the right region info,
   * so that the TransformIndexToPhysicalPoint-functions will be right.
   */
  using DummyImageType = FixedImageType;
  auto       dummyImage = DummyImageType::New();
  RegionType region;
  region.SetIndex(index);
  region.SetSize(size);
  dummyImage->SetRegions(region);
  dummyImage->SetOrigin(origin);
  dummyImage->SetSpacing(spacing);
  dummyImage->SetDirection(direction);

  /** Convert center of rotation from index-value to physical-point-value.*/
  dummyImage->TransformIndexToPhysicalPoint(centerOfRotationIndex, rotationPoint);

  /** Successfully read centerOfRotation as Index */
  return true;

} // end ReadCenterOfRotationIndex()


/**
 * ******************** ReadCenterOfRotationPoint *********************
 */

template <class TElastix>
bool
SimilarityTransformElastix<TElastix>::ReadCenterOfRotationPoint(InputPointType & rotationPoint) const
{
  /** Try to read CenterOfRotationPoint from the transform parameter
   * file, which is the rotationPoint, expressed in world coordinates.
   */
  InputPointType centerOfRotationPoint;
  bool           centerGivenAsPoint = true;
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    centerOfRotationPoint[i] = 0;

    /** Returns zero when parameter was in the parameter file. */
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

  /** copy the temporary variable into the output of this function,
   * if everything went ok.
   */
  rotationPoint = centerOfRotationPoint;

  /** Successfully read centerOfRotation as Point. */
  return true;

} // end ReadCenterOfRotationPoint()


} // end namespace elastix

#endif // end #ifndef elxSimilarityTransform_hxx
