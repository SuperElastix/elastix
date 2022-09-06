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
#ifndef elxDistancePreservingRigidityPenaltyTerm_h
#define elxDistancePreservingRigidityPenaltyTerm_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkDistancePreservingRigidityPenaltyTerm.h"

namespace elastix
{
/**
 * \class DistancePreservingRigidityPenalty
 * \brief A penalty term designed to preserve inter-voxel distances within rigid body regions.
 *
 * For more information check the paper:\n
 *  J. Kim, M. M. Matuszak, K. Saitou, and J. Balter,
 *  "Distance-preserving rigidity penalty on deformable image registration of multiple skeletal components in the neck"
 *  Medical Physics, vol. 40, no. 12, pp. 121907-1 - 121907-10, December 2013.
 * - view online: http://www.ncbi.nlm.nih.gov/pubmed/24320518
 *
 * The parameters used in this class are:
 * \parameter Metric: Select this metric as follows:\n
 *    <tt>(Metric "DistancePreservingRigidityPenalty")</tt>
 *
 * \parameter SegmentedImageName: The file name of the image to
 *    specify the rigidity index of voxels in the fixed image. The
 *  image has only non-integer values as follows:
 *    1) background: 0,
 *    2) rigid region1: 1,
 *    3) rigid region2: 2, and so on.
 *    - example: <tt>(SegmentedImageName "BoneSegmentation.mhd")</tt> \n
 *
 * \parameter PenaltyGridSpacingInVoxels: defines the grid spacing
 *  with which the rigidity penalty is calculated. In this current
 *  version, the grid spacing is set to be constant over different
 *  resolutions.
 *  - In the publication above, the grid spacing was set as [4, 4, 1].
 *
 * \author Jihun Kim, University of Michigan, Ann Arbor
 * \author Martha M. Matuszak, University of Michigan, Ann Arbor
 * \author Kazuhiro Saitou, University of Michigan, Ann Arbor
 * \author James Balter, University of Michigan, Ann Arbor
 *
 * \ingroup Metrics
 *
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT DistancePreservingRigidityPenalty
  : public itk::DistancePreservingRigidityPenaltyTerm<typename MetricBase<TElastix>::FixedImageType, double>
  , public MetricBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(DistancePreservingRigidityPenalty);

  /** Standard ITK-stuff. */
  using Self = DistancePreservingRigidityPenalty;
  using Superclass1 = itk::DistancePreservingRigidityPenaltyTerm<typename MetricBase<TElastix>::FixedImageType, double>;
  using Superclass2 = MetricBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DistancePreservingRigidityPenalty, itk::DistancePreservingRigidityPenaltyTerm);

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "DistancePreservingRigidityPenalty")</tt>\n
   */
  elxClassNameMacro("DistancePreservingRigidityPenalty");

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
  using typename Superclass1::SegmentedImageType;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Typedef's inherited from elastix. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

  /** Typedef for multi-resolution pyramid of segmented image */
  using SegmentedImagePyramidType = itk::MultiResolutionPyramidImageFilter<SegmentedImageType, SegmentedImageType>;
  using SegmentedImagePyramidPointer = typename SegmentedImagePyramidType::Pointer;

  /** Sets up a timer to measure the initialization time and
   * calls the Superclass' implementation.
   */
  void
  Initialize() override;

  /**
   * Do some things before registration:
   * \li Read the fixed rigidity image.
   * \li Setup some extra target cells.
   */
  void
  BeforeRegistration() override;

protected:
  /** The constructor. */
  DistancePreservingRigidityPenalty() = default;

  /** The destructor. */
  ~DistancePreservingRigidityPenalty() override = default;

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxDistancePreservingRigidityPenaltyTerm.hxx"
#endif

#endif // end #ifndef elxDistancePreservingRigidityPenaltyTerm_h
