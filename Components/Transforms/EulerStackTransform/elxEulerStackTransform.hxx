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
#ifndef elxEulerStackTransform_hxx
#define elxEulerStackTransform_hxx

#include "elxEulerStackTransform.h"
#include <itkDeref.h>

namespace elastix
{

/**
 * ********************* InitializeAffineTransform ****************************
 */
template <typename TElastix>
unsigned int
EulerStackTransform<TElastix>::InitializeEulerTransform()
{
  /** Initialize the m_DummySubTransform */
  m_DummySubTransform = ReducedDimensionEulerTransformType::New();
  return 0;
}


/**
 * ******************* BeforeAll ***********************
 */

template <typename TElastix>
int
EulerStackTransform<TElastix>::BeforeAll()
{
  /** Initialize affine transform. */
  return InitializeEulerTransform();
}


/**
 * ******************* BeforeRegistration ***********************
 */

template <typename TElastix>
void
EulerStackTransform<TElastix>::BeforeRegistration()
{
  /** Task 1 - Set the stack transform parameters. */

  /** Determine stack transform settings. Here they are based on the fixed image. */
  const SizeType imageSize = this->GetElastix()->GetFixedImage()->GetLargestPossibleRegion().GetSize();
  m_NumberOfSubTransforms = imageSize[SpaceDimension - 1];
  m_StackSpacing = this->GetElastix()->GetFixedImage()->GetSpacing()[SpaceDimension - 1];
  m_StackOrigin = this->GetElastix()->GetFixedImage()->GetOrigin()[SpaceDimension - 1];

  /** Set stack transform parameters. */
  m_StackTransform->SetNumberOfSubTransforms(m_NumberOfSubTransforms);
  m_StackTransform->SetStackOrigin(m_StackOrigin);
  m_StackTransform->SetStackSpacing(m_StackSpacing);

  /** Initialize stack sub transforms. */
  m_StackTransform->SetAllSubTransforms(*m_DummySubTransform);

  /** Task 2 - Give the registration an initial parameter-array. */
  this->m_Registration->GetAsITKBaseType()->SetInitialTransformParameters(
    ParametersType(this->GetNumberOfParameters(), 0.0));

  /** Task 3 - Initialize the transform */
  this->InitializeTransform();

  /** Task 4 - Set the scales. */
  this->SetScales();

} // end BeforeRegistration()


/**
 * ************************* ReadFromFile ************************
 */

template <typename TElastix>
void
EulerStackTransform<TElastix>::ReadFromFile()
{
  if (!this->HasITKTransformParameters())
  {
    const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

    /** Read stack-spacing, stack-origin and number of sub-transforms. */
    configuration.ReadParameter(m_NumberOfSubTransforms, "NumberOfSubTransforms", this->GetComponentLabel(), 0, 0);
    configuration.ReadParameter(m_StackOrigin, "StackOrigin", this->GetComponentLabel(), 0, 0);
    configuration.ReadParameter(m_StackSpacing, "StackSpacing", this->GetComponentLabel(), 0, 0);

    ReducedDimensionInputPointType RDcenterOfRotationPoint{};

    bool indexRead = false;

    /** Try first to read the CenterOfRotationPoint from the
     * transform parameter file, this is the new, and preferred
     * way, since elastix 3.402.
     */
    const bool pointRead = this->ReadCenterOfRotationPoint(RDcenterOfRotationPoint);

    if (!pointRead && !indexRead)
    {
      log::error("ERROR: No center of rotation is specified in the transform parameter file");
      itkExceptionMacro("Transform parameter file is corrupt.");
    }

    this->InitializeEulerTransform();

    m_DummySubTransform->SetCenter(RDcenterOfRotationPoint);

    if constexpr (ReducedSpaceDimension == 3)
    {
      // For 3D images, retrieve and set ComputeZYX.
      m_DummySubTransform->SetComputeZYX(
        configuration.RetrieveParameterValue(m_DummySubTransform->GetComputeZYX(), "ComputeZYX", 0, false));
    }

    /** Set stack transform parameters. */
    m_StackTransform->SetNumberOfSubTransforms(m_NumberOfSubTransforms);
    m_StackTransform->SetStackOrigin(m_StackOrigin);
    m_StackTransform->SetStackSpacing(m_StackSpacing);

    /** Set stack subtransforms. */
    m_StackTransform->SetAllSubTransforms(*m_DummySubTransform);
  }

  /** Call the ReadFromFile from the TransformBase. */
  this->Superclass2::ReadFromFile();

} // end ReadFromFile()


/**
 * ************************* CreateDerivedTransformParameterMap ************************
 */

template <typename TElastix>
auto
EulerStackTransform<TElastix>::CreateDerivedTransformParameterMap() const -> ParameterMapType
{
  const auto & itkTransform = *m_StackTransform;

  ParameterMapType parameterMap{
    { "CenterOfRotationPoint", Conversion::ToVectorOfStrings(m_DummySubTransform->GetCenter()) },
    { "StackSpacing", { Conversion::ToString(itkTransform.GetStackSpacing()) } },
    { "StackOrigin", { Conversion::ToString(itkTransform.GetStackOrigin()) } },
    { "NumberOfSubTransforms", { Conversion::ToString(itkTransform.GetNumberOfSubTransforms()) } }
  };

  if constexpr (ReducedSpaceDimension == 3)
  {
    parameterMap["ComputeZYX"] = { Conversion::ToString(m_DummySubTransform->GetComputeZYX()) };
  }

  return parameterMap;

} // end CreateDerivedTransformParameterMap()


/**
 * ********************* InitializeTransform ****************************
 */

template <typename TElastix>
void
EulerStackTransform<TElastix>::InitializeTransform()
{
  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

  /** Set all parameters to zero (no rotations, no translation). */
  m_DummySubTransform->SetIdentity();

  /** Try to read CenterOfRotationIndex from parameter file,
   * which is the rotationPoint, expressed in index-values.
   */

  ContinuousIndexType            centerOfRotationIndex{};
  InputPointType                 centerOfRotationPoint{};
  ReducedDimensionInputPointType redDimCenterOfRotationPoint{};

  const bool centerGivenAsIndex = [&configuration, &centerOfRotationIndex] {
    for (unsigned int i = 0; i < ReducedSpaceDimension; ++i)
    {
      if (!configuration.ReadParameter(centerOfRotationIndex[i], "CenterOfRotation", i, false))
      {
        return false;
      }
    }
    return true;
  }();
  const bool centerGivenAsPoint = ReadCenterOfRotationPoint(redDimCenterOfRotationPoint);
  SizeType   fixedImageSize =
    this->m_Registration->GetAsITKBaseType()->GetFixedImage()->GetLargestPossibleRegion().GetSize();

  /** Determine the center of rotation as the center of the image if no center was given */
  const bool centerGiven = centerGivenAsIndex || centerGivenAsPoint;
  if (!centerGiven)
  {
    /** Use center of image as default center of rotation */
    for (unsigned int k = 0; k < SpaceDimension; ++k)
    {
      centerOfRotationIndex[k] = (fixedImageSize[k] - 1.0) / 2.0;
    }

    /** Convert from continuous index to physical point */
    this->m_Registration->GetAsITKBaseType()->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(
      centerOfRotationIndex, centerOfRotationPoint);

    for (unsigned int k = 0; k < ReducedSpaceDimension; ++k)
    {
      redDimCenterOfRotationPoint[k] = centerOfRotationPoint[k];
    }

    /** FIX: why may the cop not work when using direction cosines? */
    bool UseDirectionCosines = true;
    configuration.ReadParameter(UseDirectionCosines, "UseDirectionCosines", 0);
    if (!UseDirectionCosines)
    {
      log::info(std::ostringstream{}
                << "warning: a wrong center of rotation could have been set,  please check the transform matrix in the "
                   "header file");
    }
  }

  /** Transform center of rotation point to physical point if given as index in parameter file. */
  if (centerGivenAsIndex)
  {
    this->m_Registration->GetAsITKBaseType()->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(
      centerOfRotationIndex, centerOfRotationPoint);

    for (unsigned int k = 0; k < ReducedSpaceDimension; ++k)
    {
      redDimCenterOfRotationPoint[k] = centerOfRotationPoint[k];
    }
  }

  /** Transform center of rotation point using initial transform if present. */
  InitialTransformCenter(redDimCenterOfRotationPoint);

  /** Set the center of rotation point. */
  m_DummySubTransform->SetCenter(redDimCenterOfRotationPoint);

  /** Set the translation to zero */
  m_DummySubTransform->SetTranslation(ReducedDimensionOutputVectorType());

  if constexpr (ReducedSpaceDimension == 3)
  {
    // For 3D images, retrieve and set ComputeZYX.
    m_DummySubTransform->SetComputeZYX(
      configuration.RetrieveParameterValue(m_DummySubTransform->GetComputeZYX(), "ComputeZYX", 0, false));
  }

  /** Set all subtransforms to a copy of the dummy Translation sub transform. */
  m_StackTransform->SetAllSubTransforms(*m_DummySubTransform);

  /** Set the initial parameters in this->m_Registration. */
  this->m_Registration->GetAsITKBaseType()->SetInitialTransformParameters(this->GetParameters());

} // end InitializeTransform()


template <typename TElastix>
void
EulerStackTransform<TElastix>::InitialTransformCenter(ReducedDimensionInputPointType & point)
{
  /** Apply the initial transform to the center of rotation, if
   * composition is used to combine the initial transform with the
   * the current (euler) transform.
   */
  if (const auto * const initialTransform = this->Superclass1::GetInitialTransform();
      initialTransform != nullptr && this->GetUseComposition())
  {
    /** Transform point to voxel coordinates. */
    InputPointType fullDimensionCenterPoint;
    for (unsigned int i = 0; i < ReducedSpaceDimension; ++i)
    {
      fullDimensionCenterPoint[i] = point[i];
    }
    fullDimensionCenterPoint[SpaceDimension - 1] = 0;
    auto fullDimensionCenterIndex =
      this->m_Registration->GetAsITKBaseType()
        ->GetFixedImage()
        ->template TransformPhysicalPointToContinuousIndex<CoordinateType>(fullDimensionCenterPoint);

    /** Get size of image and number of time points. */
    const SizeType fixedImageSize =
      this->m_Registration->GetAsITKBaseType()->GetFixedImage()->GetLargestPossibleRegion().GetSize();
    const unsigned int numTimePoints = fixedImageSize[SpaceDimension - 1];

    /** Transform center of rotation point for each time point and
     * compute average. */
    ReducedDimensionInputPointType averagePoint{};
    for (unsigned int t = 0; t < numTimePoints; ++t)
    {
      /** Set time point and transform back to point. */
      fullDimensionCenterIndex[SpaceDimension - 1] = t;
      this->m_Registration->GetAsITKBaseType()->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(
        fullDimensionCenterIndex, fullDimensionCenterPoint);

      /** Transform point using initial transform. */
      InputPointType transformedCenterOfRotationPoint = initialTransform->TransformPoint(fullDimensionCenterPoint);

      /** Add to averagePoint. */
      for (unsigned int d = 0; d < ReducedSpaceDimension; ++d)
      {
        averagePoint[d] += transformedCenterOfRotationPoint[d];
      }
    }
    for (unsigned int d = 0; d < ReducedSpaceDimension; ++d)
    {
      averagePoint[d] /= numTimePoints;
    }

    point = averagePoint;
  }
}


/**
 * ************************* SetScales *********************
 */

template <typename TElastix>
void
EulerStackTransform<TElastix>::SetScales()
{
  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

  /** Create the new scales. */
  const NumberOfParametersType numberOfParameters = this->GetNumberOfParameters();
  ScalesType                   newscales(numberOfParameters);

  /** Check if automatic scales estimation is desired. */
  bool automaticScalesEstimation = false;
  configuration.ReadParameter(automaticScalesEstimation, "AutomaticScalesEstimation", 0);

  /** Check also AutomaticScalesEstimationStackTransform for backwards compatability. */
  bool automaticScalesEstimationStackTransform = false;
  configuration.ReadParameter(
    automaticScalesEstimationStackTransform, "AutomaticScalesEstimationStackTransform", 0, false);

  if (automaticScalesEstimationStackTransform)
  {
    log::warn("WARNING: AutomaticScalesEstimationStackTransform is deprecated, use AutomaticScalesEstimation instead.");
    automaticScalesEstimation = automaticScalesEstimationStackTransform;
  }

  if (automaticScalesEstimation)
  {
    log::info("Scales are estimated automatically.");
    this->AutomaticScalesEstimationStackTransform(m_StackTransform->GetNumberOfSubTransforms(), newscales);
    log::info("finished setting scales");
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

    /** In 2D, the first parameter is an angle, the other two translations;
     * in 3D, the first three parameters are angles, the last three translations.
     */
    const unsigned int numRotationParsPerDimension = ReducedSpaceDimension == 2 ? 1 : 3;
    const unsigned int numTotalParsPerDimension = ReducedSpaceDimension == 2 ? 3 : 6;

    /** configuration.ReadParameter() returns 0 if there is a value given
     * in the parameter-file, and returns 1 if there is no value given in the
     * parameter-file.
     *
     * Check which option is used:
     * - Nothing given in the parameter-file: rotations are scaled by the default
     *   value 100000.0
     * - Only one scale given in the parameter-file: rotations are scaled by this
     *   value.
     * - All scales are given in the parameter-file: each parameter is assigned its
     *   own scale.
     */
    const double defaultScalingvalue = 10000.0;

    const int sizeLastDimension =
      this->GetElastix()->GetFixedImage()->GetLargestPossibleRegion().GetSize()[SpaceDimension - 1];

    std::size_t count = configuration.CountNumberOfParameterEntries("Scales");

    /** Check which of the above options is used. */
    if (count == 0)
    {
      /** In this case the first option is used. */
      newscales.Fill(defaultScalingvalue);

      /** The non-rotation scales are set to 1.0 for all dimensions */
      for (unsigned int i = numRotationParsPerDimension; i < (numTotalParsPerDimension * sizeLastDimension);
           i += numTotalParsPerDimension)
      {
        for (unsigned int j = numRotationParsPerDimension; j < numTotalParsPerDimension; ++j)
        {
          newscales[i + j - numRotationParsPerDimension] = 1.0;
        }
      }
    }
    else if (count == 1)
    {
      /** In this case the second option is used. */
      double scale = defaultScalingvalue;
      configuration.ReadParameter(scale, "Scales", 0);
      newscales.Fill(scale);

      /** The non-rotation scales are set to 1.0 for all dimensions */
      for (unsigned int i = numRotationParsPerDimension; i < (numTotalParsPerDimension * sizeLastDimension);
           i += numTotalParsPerDimension)
      {
        for (unsigned int j = numRotationParsPerDimension; j < numTotalParsPerDimension; ++j)
        {
          newscales[i + j - numRotationParsPerDimension] = 1.0;
        }
      }
    }
    else if (count == numberOfParameters)
    {
      newscales.Fill(1.0);
      /** In this case the third option is used. */
      for (unsigned int i = 0; i < numberOfParameters; ++i)
      {
        configuration.ReadParameter(newscales[i], "Scales", i);
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

  /** And set the scales into the optimizer. */
  this->m_Registration->GetAsITKBaseType()->GetModifiableOptimizer()->SetScales(newscales);

} // end SetScales()


/**
 * ******************** ReadCenterOfRotationPoint *********************
 */

template <typename TElastix>
bool
EulerStackTransform<TElastix>::ReadCenterOfRotationPoint(ReducedDimensionInputPointType & rotationPoint) const
{
  /** Try to read CenterOfRotationPoint from the transform parameter
   * file, which is the rotationPoint, expressed in world coordinates.
   */
  ReducedDimensionInputPointType redDimCenterOfRotationPoint{};
  for (unsigned int i = 0; i < ReducedSpaceDimension; ++i)
  {
    /** Returns zero when parameter was in the parameter file. */
    bool found =
      this->m_Configuration->ReadParameter(redDimCenterOfRotationPoint[i], "CenterOfRotationPoint", i, false);
    if (!found)
    {
      return false;
    }
  }

  /** copy the temporary variable into the output of this function,
   * if everything went ok.
   */
  rotationPoint = redDimCenterOfRotationPoint;

  /** Successfully read centerOfRotation as Point. */
  return true;

} // end ReadCenterOfRotationPoint()


} // end namespace elastix

#endif // end #ifndef elxEulerStackTransform_hxx
