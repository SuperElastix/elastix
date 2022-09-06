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
#ifndef elxTransformRigidityPenaltyTerm_h
#define elxTransformRigidityPenaltyTerm_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkTransformRigidityPenaltyTerm.h"

namespace elastix
{

/**
 * \class TransformRigidityPenalty
 * \brief A penalty term based on non-rigidity.
 *
 * For more information check the paper:\n
 * M. Staring, S. Klein and J.P.W. Pluim,
 * "A Rigidity Penalty Term for Nonrigid Registration,"
 * Medical Physics, vol. 34, no. 11, pp. 4098 - 4108, November 2007.
 *
 * The parameters used in this class are:
 * \parameter Metric: Select this metric as follows:\n
 *    <tt>(Metric "TransformRigidityPenalty")</tt>
 *
 * \parameter LinearityConditionWeight: A parameter to weigh the linearity
 *    condition term of the rigidity term. \n
 *    example: <tt>(LinearityConditionWeight 2.0)</tt> \n
 *    Default is 1.0.
 * \parameter OrthonormalityConditionWeight: A parameter to weigh the
 *    orthonormality condition term of the rigidity term. \n
 *    example: <tt>(OrthonormalityConditionWeight 2.0)</tt> \n
 *    Default is 1.0.
 * \parameter PropernessConditionWeight: A parameter to weigh the properness
 *    condition term of the rigidity term. \n
 *    example: <tt>(PropernessConditionWeight 2.0)</tt> \n
 *    Default is 1.0.
 * \parameter UseLinearityCondition: A flag to specify the usage of the
 *    linearity condition term for optimisation. \n
 *    example: <tt>(UseLinearityCondition "false")</tt> \n
 *    Default is "true".
 * \parameter UseOrthonormalityCondition: A flag to specify the usage of
 *    the orthonormality condition term for optimisation. \n
 *    example: <tt>(UseOrthonormalityCondition "false")</tt> \n
 *    Default is "true".
 * \parameter UsePropernessCondition: A flag to specify the usage of the
 *    properness condition term for optimisation. \n
 *    example: <tt>(UsePropernessCondition "false")</tt> \n
 *    Default is "true".
 * \parameter CalculateLinearityCondition: A flag to specify if the linearity
 *    condition should still be calculated, even if it is not used for
 *    optimisation. \n
 *    example: <tt>(CalculateLinearityCondition "false")</tt> \n
 *    Default is "true".
 * \parameter CalculateOrthonormalityCondition: A flag to specify if the
 *    orthonormality condition should still be calculated, even if it is
 *    not used for optimisation. \n
 *    example: <tt>(CalculateOrthonormalityCondition "false")</tt> \n
 *    Default is "true".
 * \parameter CalculatePropernessCondition: A flag to specify if the properness
 *    condition should still be calculated, even if it is not used for
 *    optimisation. \n
 *    example: <tt>(CalculatePropernessCondition "false")</tt> \n
 *    Default is "true".
 * \parameter FixedRigidityImageName: the name of a coefficient image to
 *    specify the rigidity index of voxels in the fixed image. \n
 *    example: <tt>(FixedRigidityImageName "fixedRigidityImage.mhd")</tt> \n
 *    If not supplied the rigidity coefficient is not based on the fixed
 *    image, which is recommended.\n
 *    If neither FixedRigidityImageName nor MovingRigidityImageName are
 *    supplied, the rigidity penalty term is evaluated on the whole transform
 *    input domain.
 * \parameter MovingRigidityImageName: the name of a coefficient image to
 *    specify the rigidity index of voxels in the moving image. \n
 *    example: <tt>(MovingRigidityImageName "movingRigidityImage.mhd")</tt> \n
 *    If not supplied the rigidity coefficient is not based on the moving
 *    image, which is NOT recommended.\n
 *    If neither FixedRigidityImageName nor MovingRigidityImageName are
 *    supplied, the rigidity penalty term is evaluated on the whole transform
 *    input domain.
 * \parameter DilateRigidityImages: flag to specify the dilation of the
 *    rigidity coefficient images. With this the region of rigidity can be
 *    extended to force rigidity of the inner region. \n
 *    example: <tt>(DilateRigidityImages "false" "false" "true")</tt> \n
 *    Default is "true".
 * \parameter DilationRadiusMultiplier: the dilation radius is a multiplier
 *    times the grid spacing of the B-spline transform. \n
 *    example: <tt>(DilationRadiusMultiplier 1.0 1.0 2.0)</tt> \n
 *    Default is 1.0.
 *
 * \ingroup Metrics
 *
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT TransformRigidityPenalty
  : public itk::TransformRigidityPenaltyTerm<typename MetricBase<TElastix>::FixedImageType, double>
  , public MetricBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(TransformRigidityPenalty);

  /** Standard ITK-stuff. */
  using Self = TransformRigidityPenalty;
  using Superclass1 = itk::TransformRigidityPenaltyTerm<typename MetricBase<TElastix>::FixedImageType, double>;
  using Superclass2 = MetricBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TransformRigidityPenalty, itk::TransformRigidityPenaltyTerm);

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "TransformRigidityPenalty")</tt>\n
   */
  elxClassNameMacro("TransformRigidityPenalty");

  /** Typedefs from the superclass. */
  using typename Superclass1::CoordinateRepresentationType;
  using typename Superclass1::MovingImageType;
  using typename Superclass1::MovingImagePixelType;
  using typename Superclass1::MovingImageConstPointer;
  using typename Superclass1::FixedImageType;
  using typename Superclass1::FixedImageConstPointer;
  using typename Superclass1::FixedImageRegionType;
  using typename Superclass1::TransformType;
  using typename Superclass1::TransformPointer;
  using typename Superclass1::InputPointType;
  using typename Superclass1::OutputPointType;
  using typename Superclass1::TransformParametersType;
  using typename Superclass1::TransformJacobianType;
  using typename Superclass1::InterpolatorType;
  using typename Superclass1::InterpolatorPointer;
  using typename Superclass1::RealType;
  using typename Superclass1::GradientPixelType;
  using typename Superclass1::GradientImageType;
  using typename Superclass1::GradientImagePointer;
  using typename Superclass1::GradientImageFilterType;
  using typename Superclass1::GradientImageFilterPointer;
  using typename Superclass1::FixedImageMaskType;
  using typename Superclass1::FixedImageMaskPointer;
  using typename Superclass1::MovingImageMaskType;
  using typename Superclass1::MovingImageMaskPointer;
  using typename Superclass1::MeasureType;
  using typename Superclass1::DerivativeType;
  using typename Superclass1::ParametersType;
  using typename Superclass1::FixedImagePixelType;
  using typename Superclass1::MovingImageRegionType;
  using typename Superclass1::ImageSamplerType;
  using typename Superclass1::ImageSamplerPointer;
  using typename Superclass1::ImageSampleContainerType;
  using typename Superclass1::ImageSampleContainerPointer;
  using typename Superclass1::FixedImageLimiterType;
  using typename Superclass1::MovingImageLimiterType;
  using typename Superclass1::FixedImageLimiterOutputType;
  using typename Superclass1::MovingImageLimiterOutputType;

  using typename Superclass1::CoefficientImageType;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Typedef's inherited from elastix. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

  /** Sets up a timer to measure the initialization time and
   * calls the Superclass' implementation.
   */
  void
  Initialize() override;

  /**
   * Do some things before each resolution:
   * \li Read all parameters.
   */
  void
  BeforeEachResolution() override;

  /**
   * Do some things before registration:
   * \li Read the fixed rigidity image.
   * \li Read the moving rigidity image.
   * \li Setup some extra target cells.
   */
  void
  BeforeRegistration() override;

  /**
   * Do some things after each iteration:
   * \li Print the OC, PC, LC parts of the rigidity term.
   */
  void
  AfterEachIteration() override;

  /** This metric is advanced (so it has a sampling possibility), but it
   * purposely does not use samplers. The MetricBase class, however, issues
   * a warning if this is the case, so we overwrite that function.
   */
  void
  SelectNewSamples() override
  {}

protected:
  /** The constructor. */
  TransformRigidityPenalty() = default;

  /** The destructor. */
  ~TransformRigidityPenalty() override = default;

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxTransformRigidityPenaltyTerm.hxx"
#endif

#endif // end #ifndef elxTransformRigidityPenaltyTerm_h
