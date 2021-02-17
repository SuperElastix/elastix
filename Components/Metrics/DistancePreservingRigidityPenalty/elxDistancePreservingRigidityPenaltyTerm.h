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
  /** Standard ITK-stuff. */
  typedef DistancePreservingRigidityPenalty                                                                 Self;
  typedef itk::DistancePreservingRigidityPenaltyTerm<typename MetricBase<TElastix>::FixedImageType, double> Superclass1;
  typedef MetricBase<TElastix>                                                                              Superclass2;
  typedef itk::SmartPointer<Self>                                                                           Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

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
  typedef typename Superclass1::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass1::MovingImageType              MovingImageType;
  typedef typename Superclass1::MovingImagePixelType         MovingImagePixelType;
  typedef typename Superclass1::MovingImageConstPointer      MovingImageConstPointer;
  typedef typename Superclass1::FixedImageType               FixedImageType;
  typedef typename Superclass1::FixedImageConstPointer       FixedImageConstPointer;
  typedef typename Superclass1::FixedImageRegionType         FixedImageRegionType;
  typedef typename Superclass1::TransformType                TransformType;
  typedef typename Superclass1::TransformPointer             TransformPointer;
  typedef typename Superclass1::InputPointType               InputPointType;
  typedef typename Superclass1::OutputPointType              OutputPointType;
  typedef typename Superclass1::TransformParametersType      TransformParametersType;
  typedef typename Superclass1::TransformJacobianType        TransformJacobianType;
  typedef typename Superclass1::InterpolatorType             InterpolatorType;
  typedef typename Superclass1::InterpolatorPointer          InterpolatorPointer;
  typedef typename Superclass1::RealType                     RealType;
  typedef typename Superclass1::GradientPixelType            GradientPixelType;
  typedef typename Superclass1::GradientImageType            GradientImageType;
  typedef typename Superclass1::GradientImagePointer         GradientImagePointer;
  typedef typename Superclass1::GradientImageFilterType      GradientImageFilterType;
  typedef typename Superclass1::GradientImageFilterPointer   GradientImageFilterPointer;
  typedef typename Superclass1::FixedImageMaskType           FixedImageMaskType;
  typedef typename Superclass1::FixedImageMaskPointer        FixedImageMaskPointer;
  typedef typename Superclass1::MovingImageMaskType          MovingImageMaskType;
  typedef typename Superclass1::MovingImageMaskPointer       MovingImageMaskPointer;
  typedef typename Superclass1::MeasureType                  MeasureType;
  typedef typename Superclass1::DerivativeType               DerivativeType;
  typedef typename Superclass1::ParametersType               ParametersType;
  typedef typename Superclass1::FixedImagePixelType          FixedImagePixelType;
  typedef typename Superclass1::MovingImageRegionType        MovingImageRegionType;
  typedef typename Superclass1::ImageSamplerType             ImageSamplerType;
  typedef typename Superclass1::ImageSamplerPointer          ImageSamplerPointer;
  typedef typename Superclass1::ImageSampleContainerType     ImageSampleContainerType;
  typedef typename Superclass1::ImageSampleContainerPointer  ImageSampleContainerPointer;
  typedef typename Superclass1::FixedImageLimiterType        FixedImageLimiterType;
  typedef typename Superclass1::MovingImageLimiterType       MovingImageLimiterType;
  typedef typename Superclass1::FixedImageLimiterOutputType  FixedImageLimiterOutputType;
  typedef typename Superclass1::MovingImageLimiterOutputType MovingImageLimiterOutputType;
  typedef typename Superclass1::CoefficientImageType         CoefficientImageType;
  typedef typename Superclass1::SegmentedImageType           SegmentedImageType;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Typedef's inherited from elastix. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

  /** Typedef for multi-resolution pyramid of segmented image */
  typedef itk::MultiResolutionPyramidImageFilter<SegmentedImageType, SegmentedImageType> SegmentedImagePyramidType;
  typedef typename SegmentedImagePyramidType::Pointer                                    SegmentedImagePyramidPointer;

  /** Sets up a timer to measure the initialization time and
   * calls the Superclass' implementation.
   */
  void
  Initialize(void) override;

  /**
   * Do some things before registration:
   * \li Read the fixed rigidity image.
   * \li Setup some extra target cells.
   */
  void
  BeforeRegistration(void) override;

protected:
  /** The constructor. */
  DistancePreservingRigidityPenalty() = default;

  /** The destructor. */
  ~DistancePreservingRigidityPenalty() override = default;

private:
  elxOverrideGetSelfMacro;

  /** The deleted copy constructor. */
  DistancePreservingRigidityPenalty(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxDistancePreservingRigidityPenaltyTerm.hxx"
#endif

#endif // end #ifndef elxDistancePreservingRigidityPenaltyTerm_h
