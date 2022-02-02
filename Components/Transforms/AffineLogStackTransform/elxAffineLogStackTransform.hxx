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

#ifndef ELXAFFINELOGSTACKTRANSFORM_HXX
#define ELXAFFINELOGSTACKTRANSFORM_HXX
#include "elxAffineLogStackTransform.h"

#include "itkImageRegionExclusionConstIteratorWithIndex.h"
#include <vnl/vnl_math.h>

namespace elastix
{

/**
 * ********************* InitializeAffineTransform ****************************
 */
template <class TElastix>
unsigned int
AffineLogStackTransform<TElastix>::InitializeAffineLogTransform()
{
  /** Initialize the m_DummySubTransform */
  this->m_DummySubTransform = ReducedDimensionAffineLogTransformBaseType::New();

  return 0;
}


/**
 * ******************* BeforeAll ***********************
 */

template <class TElastix>
int
AffineLogStackTransform<TElastix>::BeforeAll()
{
  /** Initialize affine transform. */
  return InitializeAffineLogTransform();
}


/**
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
void
AffineLogStackTransform<TElastix>::BeforeRegistration()
{
  /** Task 1 - Set the stack transform parameters. */

  /** Determine stack transform settings. Here they are based on the fixed image. */
  const SizeType imageSize = this->GetElastix()->GetFixedImage()->GetLargestPossibleRegion().GetSize();
  this->m_NumberOfSubTransforms = imageSize[SpaceDimension - 1];
  this->m_StackSpacing = this->GetElastix()->GetFixedImage()->GetSpacing()[SpaceDimension - 1];
  this->m_StackOrigin = this->GetElastix()->GetFixedImage()->GetOrigin()[SpaceDimension - 1];

  /** Set stack transform parameters. */
  this->m_StackTransform->SetNumberOfSubTransforms(this->m_NumberOfSubTransforms);
  this->m_StackTransform->SetStackOrigin(this->m_StackOrigin);
  this->m_StackTransform->SetStackSpacing(this->m_StackSpacing);

  /** Initialize stack sub transforms. */
  this->m_StackTransform->SetAllSubTransforms(*m_DummySubTransform);

  /** Task 3 - Give the registration an initial parameter-array. */
  this->m_Registration->GetAsITKBaseType()->SetInitialTransformParameters(
    ParametersType(this->GetNumberOfParameters(), 0.0));

  /** Task 4 - Initialize the transform */
  this->InitializeTransform();

  /** Task 2 - Set the scales. */
  this->SetScales();

} // end BeforeRegistration()


/**
 * ************************* ReadFromFile ************************
 */

template <class TElastix>
void
AffineLogStackTransform<TElastix>::ReadFromFile()
{
  if (!this->HasITKTransformParameters())
  {
    /** Read stack-spacing, stack-origin and number of sub-transforms. */
    this->GetConfiguration()->ReadParameter(
      this->m_NumberOfSubTransforms, "NumberOfSubTransforms", this->GetComponentLabel(), 0, 0);
    this->GetConfiguration()->ReadParameter(this->m_StackOrigin, "StackOrigin", this->GetComponentLabel(), 0, 0);
    this->GetConfiguration()->ReadParameter(this->m_StackSpacing, "StackSpacing", this->GetComponentLabel(), 0, 0);

    ReducedDimensionInputPointType RDcenterOfRotationPoint{};

    /** Try first to read the CenterOfRotationPoint from the
     * transform parameter file, this is the new, and preferred
     * way, since elastix 3.402.
     */
    const bool pointRead = this->ReadCenterOfRotationPoint(RDcenterOfRotationPoint);

    if (!pointRead)
    {
      xl::xout["error"] << "ERROR: No center of rotation is specified in the transform parameter file" << std::endl;
      itkExceptionMacro(<< "Transform parameter file is corrupt.")
    }

    this->InitializeAffineLogTransform();

    this->m_DummySubTransform->SetCenter(RDcenterOfRotationPoint);

    /** Set stack transform parameters. */
    this->m_StackTransform->SetNumberOfSubTransforms(this->m_NumberOfSubTransforms);
    this->m_StackTransform->SetStackOrigin(this->m_StackOrigin);
    this->m_StackTransform->SetStackSpacing(this->m_StackSpacing);

    /** Set stack subtransforms. */
    this->m_StackTransform->SetAllSubTransforms(*m_DummySubTransform);
  }

  /** Call the ReadFromFile from the TransformBase. */
  this->Superclass2::ReadFromFile();

} // end ReadFromFile()


/**
 * ************************* CreateDerivedTransformParametersMap ************************
 */

template <class TElastix>
auto
AffineLogStackTransform<TElastix>::CreateDerivedTransformParametersMap() const -> ParameterMapType
{
  const auto & itkTransform = *m_StackTransform;

  return { { "CenterOfRotationPoint", Conversion::ToVectorOfStrings(m_DummySubTransform->GetCenter()) },
           { "StackSpacing", { Conversion::ToString(itkTransform.GetStackSpacing()) } },
           { "StackOrigin", { Conversion::ToString(itkTransform.GetStackOrigin()) } },
           { "NumberOfSubTransforms", { Conversion::ToString(itkTransform.GetNumberOfSubTransforms()) } } };

} // end CreateDerivedTransformParametersMap()


/**
 * ********************* InitializeTransform ****************************
 */

template <class TElastix>
void
AffineLogStackTransform<TElastix>::InitializeTransform()
{
  /** Set all parameters to zero (no rotations, no translation). */
  this->m_DummySubTransform->SetIdentity();

  /** Try to read CenterOfRotationIndex from parameter file,
   * which is the rotationPoint, expressed in index-values.
   */

  ContinuousIndexType                 centerOfRotationIndex{};
  InputPointType                      centerOfRotationPoint{};
  ReducedDimensionContinuousIndexType RDcenterOfRotationIndex{};
  ReducedDimensionInputPointType      RDcenterOfRotationPoint{};
  InputPointType                      TransformedCenterOfRotation{};
  ReducedDimensionInputPointType      RDTransformedCenterOfRotation{};

  bool     centerGivenAsIndex = true;
  bool     centerGivenAsPoint = true;
  SizeType fixedImageSize =
    this->m_Registration->GetAsITKBaseType()->GetFixedImage()->GetLargestPossibleRegion().GetSize();

  for (unsigned int i = 0; i < ReducedSpaceDimension; ++i)
  {
    /** Check COR index: Returns zero when parameter was in the parameter file. */
    bool foundI = this->m_Configuration->ReadParameter(centerOfRotationIndex[i], "CenterOfRotation", i, false);
    if (!foundI)
    {
      centerGivenAsIndex = false;
    }

    /** Check COR point: Returns zero when parameter was in the parameter file. */
    bool foundP = this->m_Configuration->ReadParameter(RDcenterOfRotationPoint[i], "CenterOfRotationPoint", i, false);
    if (!foundP)
    {
      centerGivenAsPoint = false;
    }
  } // end loop over SpaceDimension

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

  /** Set the center of rotation to the center of the image if no center was given */
  bool centerGiven = centerGivenAsIndex || centerGivenAsPoint;
  if (!centerGiven)
  {
    /** Use center of image as default center of rotation */
    for (unsigned int k = 0; k < SpaceDimension; ++k)
    {
      centerOfRotationIndex[k] = (fixedImageSize[k] - 1.0) / 2.0;
    }
    /** Convert from continuous index to physical point */
    this->m_Registration->GetAsITKBaseType()->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(
      centerOfRotationIndex, TransformedCenterOfRotation);

    for (unsigned int k = 0; k < ReducedSpaceDimension; ++k)
    {
      RDTransformedCenterOfRotation[k] = TransformedCenterOfRotation[k];
    }

    this->m_DummySubTransform->SetCenter(RDTransformedCenterOfRotation);
  }

  /** Set the center of rotation if it was entered by the user. */
  if (centerGivenAsPoint)
  {
    this->m_DummySubTransform->SetCenter(RDcenterOfRotationPoint);
  }
  if (centerGivenAsIndex)
  {
    this->m_Registration->GetAsITKBaseType()->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(
      centerOfRotationIndex, TransformedCenterOfRotation);
    for (unsigned int k = 0; k < ReducedSpaceDimension; ++k)
    {
      RDTransformedCenterOfRotation[k] = TransformedCenterOfRotation[k];
    }
    this->m_DummySubTransform->SetCenter(RDTransformedCenterOfRotation);
  }

  /** Set the translation to zero */
  this->m_DummySubTransform->SetTranslation(ReducedDimensionOutputVectorType());

  /** Set all subtransforms to a copy of the dummy Translation sub transform. */
  this->m_StackTransform->SetAllSubTransforms(*m_DummySubTransform);

  /** Set the initial parameters in this->m_Registration. */
  this->m_Registration->GetAsITKBaseType()->SetInitialTransformParameters(this->GetParameters());

} // end InitializeTransform()


/**
 * ************************* SetScales *********************
 */

template <class TElastix>
void
AffineLogStackTransform<TElastix>::SetScales()
{
  /** Create the new scales. */
  const NumberOfParametersType numberOfParameters = this->GetNumberOfParameters();
  ScalesType                   newscales(numberOfParameters);

  /** Check if automatic scales estimation is desired. */
  bool automaticScalesEstimation = false;
  this->m_Configuration->ReadParameter(automaticScalesEstimation, "AutomaticScalesEstimation", 0);

  /** Check also AutomaticScalesEstimationStackTransform for backwards compatability. */
  bool automaticScalesEstimationStackTransform = false;
  this->m_Configuration->ReadParameter(
    automaticScalesEstimationStackTransform, "AutomaticScalesEstimationStackTransform", 0, false);

  if (automaticScalesEstimationStackTransform)
  {
    xl::xout["warning"]
      << "WARNING: AutomaticScalesEstimationStackTransform is deprecated, use AutomaticScalesEstimation instead."
      << std::endl;
    automaticScalesEstimation = automaticScalesEstimationStackTransform;
  }

  if (automaticScalesEstimation)
  {
    elxout << "Scales are estimated automatically." << std::endl;
    this->AutomaticScalesEstimationStackTransform(this->m_StackTransform->GetNumberOfSubTransforms(), newscales);
    elxout << "finished setting scales" << std::endl;
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

    /** The first SpaceDimension * SpaceDimension number of parameters
     * represent rotations (4 in 2D and 9 in 3D).
     */

    const unsigned int rotationPart = (ReducedSpaceDimension) * (ReducedSpaceDimension);
    const unsigned int totalPart = (SpaceDimension) * (ReducedSpaceDimension);

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
    const double defaultScalingvalue = 10000.0;

    int sizeLastDimension =
      this->GetElastix()->GetFixedImage()->GetLargestPossibleRegion().GetSize()[SpaceDimension - 1];

    std::size_t count = this->m_Configuration->CountNumberOfParameterEntries("Scales");

    /** Check which of the above options is used. */
    if (count == 0)
    {
      /** In this case the first option is used. */
      newscales.Fill(defaultScalingvalue);

      /** The non-rotation scales are set to 1.0 */
      for (unsigned int i = rotationPart; i < (totalPart * sizeLastDimension); i = i + totalPart)
      {
        newscales[i] = 1.0;
        newscales[i + 1] = 1.0;
      }
    }

    else if (count == 1)
    {
      /** In this case the second option is used. */
      double scale = defaultScalingvalue;
      this->m_Configuration->ReadParameter(scale, "Scales", 0);
      newscales.Fill(scale);

      /** The non-rotation scales are set to 1.0 */
      for (unsigned int i = rotationPart; i < (totalPart * sizeLastDimension); i = i + totalPart)
      {
        newscales[i] = 1.0;
        newscales[i + 1] = 1.0;
      }
    }
    else if (count == numberOfParameters)
    {
      newscales.Fill(1.0);
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
      itkExceptionMacro(<< "ERROR: The Scales-option in the parameter-file has not been set properly.");
    }

  } // end else: no automaticScalesEstimation

  elxout << "Scales for transform parameters are: " << newscales << std::endl;

  /** And set the scales into the optimizer. */
  this->m_Registration->GetAsITKBaseType()->GetModifiableOptimizer()->SetScales(newscales);

} // end SetScales()


/**
 * ******************** ReadCenterOfRotationPoint *********************
 */

template <class TElastix>
bool
AffineLogStackTransform<TElastix>::ReadCenterOfRotationPoint(ReducedDimensionInputPointType & rotationPoint) const
{
  /** Try to read CenterOfRotationPoint from the transform parameter
   * file, which is the rotationPoint, expressed in world coordinates.
   */
  ReducedDimensionInputPointType RDcenterOfRotationPoint{};
  bool                           centerGivenAsPoint = true;
  for (unsigned int i = 0; i < ReducedSpaceDimension; ++i)
  {
    /** Returns zero when parameter was in the parameter file. */
    bool found = this->m_Configuration->ReadParameter(RDcenterOfRotationPoint[i], "CenterOfRotationPoint", i, false);
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
  rotationPoint = RDcenterOfRotationPoint;

  /** Successfully read centerOfRotation as Point. */
  return true;

} // end ReadCenterOfRotationPoint()


} // end namespace elastix

#endif // ELXAFFINELOGSTACKTRANSFORM_HXX
