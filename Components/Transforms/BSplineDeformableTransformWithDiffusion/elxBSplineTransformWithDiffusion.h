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
#ifndef elxBSplineTransformWithDiffusion_h
#define elxBSplineTransformWithDiffusion_h

/* For easy changing the BSplineOrder: */
#define __VSplineOrder 3

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkAdvancedCombinationTransform.h"
//#include "itkBSplineCombinationTransform.h"
#include "itkBSplineResampleImageFilterBase.h"
#include "itkBSplineUpsampleImageFilter.h"

#include "itkImageRegionConstIterator.h"

/** Include structure for the diffusion. */
#include "itkDeformationFieldRegulizer.h"
#include "itkVectorMeanDiffusionImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkMaximumImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkBSplineInterpolateImageFunction.h"

namespace elastix
{

/**
 * \class BSplineTransformWithDiffusion
 * \brief This class combines a B-spline transform with the
 * diffusion/filtering of the deformation field.
 *
 * Every n iterations the deformation field is diffused using the
 * VectorMeanDiffusionImageFilter. The total transformation of a point
 * is determined by adding the B-spline deformation to the
 * deformation field arrow. Filtering of the deformation field is based
 * on some 'stiffness coefficient' image.
 *
 * \todo: this Transform has not been tested for images with Direction cosines
 * matrix other than the identity matrix.
 *
 * \warning Using this transform in with the option
 * (HowToCombineTranforms "Compose"). May give unpredictable results.
 * Especially if the initial transformation is large. This is because
 * the coefficient grid is not properly initialized.
 * Better use (HowToCombineTranforms "Add").
 *
 * The parameters used in this class are:
 * \parameter Transform: Select this transform as follows:\n
 *    <tt>(%Transform "BSplineTransformWithDiffusion")</tt>
 * \parameter FinalGridSpacing: the grid spacing of the B-spline transform
 *    part of this transform for each dimension. \n
 *    example: <tt>(FinalGridSpacing 8.0 8.0 8.0)</tt>
 *    If only one argument is given, that factor is used for each dimension. The spacing
 *    is not in millimeters, but in "voxel size units".
 *    The default is 8.0 in every dimension.
 * \parameter UpsampleGridOption: whether or not the B-spline grid should
 *    be upsampled from one resolution level to another. Choose from {true, false}. \n
 *    example: <tt>(UpsampleGridOption "true")</tt>
 *    The default is "true".
 * \parameter FilterPattern: defines according to what schedule/pattern
 *    the deformation field should be filtered. Choose from {1,2}, where
 *    FilterPattern 1 is diffusion every "DiffusionEachNIterations" iterations,
 *    and where FilterPattern 2 filters more frequent in the beginning
 *    and less frequent at the end of a resolution. \n
 *    example: <tt>(FilterPattern 1)</tt>
 *    The default is filter pattern 1.
 * \parameter DiffusionEachNIterations: defines for FilterPattern 1 after how
 *    many iterations of the optimiser there should be a filtering step. \n
 *    example: <tt>(DiffusionEachNIterations 5)</tt>
 *    The default is 1.
 * \parameter AfterIterations: defines for FilterPattern 2 after how many
 *    iterations of the optimiser the filter frequency should be increased. \n
 *    example: <tt>(AfterIterations 100 200)</tt>
 *    The default is 50 and 100.
 * \parameter HowManyIterations: defines to what frequency the filtering
 *    should be increased. \n
 *    example: <tt>(HowManyIterations 1 5 10)</tt>
 *    The default is 1, 5 and 10.
 * \parameter NumberOfDiffusionIterations: defines the number of times
 *    the adaptive filtering is performed. \n
 *    example: <tt>(NumberOfDiffusionIterations 10)</tt>
 *    The default is 1.
 * \parameter Radius: defines the radius of the filter. \n
 *    example: <tt>(Radius 1)</tt>
 *    The default is 1.
 * \parameter ThresholdBool: defines whether or not the stiffness coefficient
 *    image should be thresholded. Choose from {true, false}. \n
 *    example: <tt>(ThresholdBool "true")</tt>
 *    The default is "true".
 * \parameter ThresholdHU: if it is thresholded, this defines the threshold in
 *    Houndsfield units. \n
 *    example: <tt>(ThresholdHU 150)</tt>
 *    The default is 150.
 * \parameter WriteDiffusionFiles: defines whether or not the stiffness coefficient
 *    image, the deformation field and the filtered field should be written
 *    to file. Choose from {true, false}. \n
 *    example: <tt>(WriteDiffusionFiles "true")</tt>
 *    The default is "false".
 * \parameter GrayValueImageAlsoBasedOnFixedImage: defines whether or not
 *    the stiffness coefficient image should also be based on the fixed image.
 *    Choose from {true, false}. \n
 *    example: <tt>(GrayValueImageAlsoBasedOnFixedImage "true")</tt>
 *    The default is "true".
 * \parameter UseFixedSegmentation: defines whether or not the stiffness coefficient
 *    image should be based on some prior defined segmentation of rigid structures in
 *    the fixed image, instead on a thresholding. Choose from {true, false}. \n
 *    example: <tt>(UseFixedSegmentation "true")</tt>
 *    The default is "false".
 * \parameter FixedSegmentationFileName: the filename of this segmentation. \n
 *    example: <tt>(FixedSegmentationFileName "somestring")</tt>
 * \parameter UseMovingSegmentation: defines whether or not the stiffness coefficient
 *    image should be based on some prior defined segmentation of rigid structures in
 *    the moving image, instead on a thresholding. Choose from {true, false}. \n
 *    example: <tt>(UseMovingSegmentation "true")</tt>
 *    The default is "false".
 * \parameter MovingSegmentationFileName: the filename of this segmentation. \n
 *    example: <tt>(MovingSegmentationFileName "somestring")</tt>
 * \parameter DefaultPixelValueForGVI: the default pixel value, when resampling
 *    the grayvalueimage.
 *    example: <tt>(DefaultPixelValueForGVI 0)</tt>
 *    The default is 0.
 *
 * The transform parameters necessary for transformix, additionally defined by this class, are:
 * \transformparameter DeformationFieldFileName: stores the name of the deformation field. \n
 *    example: <tt>(DeformationFieldFileName "defField.mhd")</tt>
 * \transformparameter GridSize: stores the size of the B-spline grid. \n
 *    example: <tt>(GridSize 16 16 16)</tt>
 * \transformparameter GridIndex: stores the index of the B-spline grid. \n
 *    example: <tt>(GridIndex 0 0 0)</tt>
 * \transformparameter GridSpacing: stores the spacing of the B-spline grid. \n
 *    example: <tt>(GridSpacing 16.0 16.0 16.0)</tt>
 * \transformparameter GridOrigin: stores the origin of the B-spline grid. \n
 *    example: <tt>(GridOrigin 0.0 0.0 0.0)</tt>
 *
 * \ingroup Transforms
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT BSplineTransformWithDiffusion
  : public itk::DeformationFieldRegulizer<itk::AdvancedCombinationTransform<
      // BSplineCombinationTransform<
      typename elx::TransformBase<TElastix>::CoordRepType,
      elx::TransformBase<TElastix>::FixedImageDimension>>
  ,
    // elx::TransformBase<TElastix>::FixedImageDimension, __VSplineOrder > >,
    public TransformBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(BSplineTransformWithDiffusion);

  /** Standard ITK-stuff. */
  using Self = BSplineTransformWithDiffusion;
  using Superclass1 = itk::DeformationFieldRegulizer<
    itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                      elx::TransformBase<TElastix>::FixedImageDimension>>;
  using Superclass2 = elx::TransformBase<TElastix>;

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform.
   */
  using BSplineTransformType =
    itk::AdvancedBSplineDeformableTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                            elx::TransformBase<TElastix>::FixedImageDimension,
                                            3>;

  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BSplineTransformWithDiffusion, itk::DeformationFieldRegulizer);

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "BSplineTransformWithDiffusion")</tt>\n
   */
  elxClassNameMacro("BSplineTransformWithDiffusion");

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass2::FixedImageDimension);

  /** The B-spline order. */
  itkStaticConstMacro(SplineOrder, unsigned int, __VSplineOrder);

  /** Typedefs inherited from the superclass. */
  using typename Superclass1::ScalarType;
  using typename Superclass1::ParametersType;
  using typename Superclass1::JacobianType;
  using typename Superclass1::InputVectorType;
  using typename Superclass1::OutputVectorType;
  using typename Superclass1::InputCovariantVectorType;
  using typename Superclass1::OutputCovariantVectorType;
  using typename Superclass1::InputVnlVectorType;
  using typename Superclass1::OutputVnlVectorType;
  using typename Superclass1::InputPointType;
  using typename Superclass1::OutputPointType;

  /** Typedef's specific for the BSplineTransform. */
  using PixelType = typename BSplineTransformType::PixelType;
  using ImageType = typename BSplineTransformType::ImageType;
  using ImagePointer = typename BSplineTransformType::ImagePointer;
  using RegionType = typename BSplineTransformType::RegionType;
  using IndexType = typename BSplineTransformType::IndexType;
  using SizeType = typename BSplineTransformType::SizeType;
  using SpacingType = typename BSplineTransformType::SpacingType;
  using OriginType = typename BSplineTransformType::OriginType;
  using WeightsFunctionType = typename BSplineTransformType::WeightsFunctionType;
  using WeightsType = typename BSplineTransformType::WeightsType;
  using ContinuousIndexType = typename BSplineTransformType::ContinuousIndexType;
  using ParameterIndexArrayType = typename BSplineTransformType::ParameterIndexArrayType;

  /** Typedef's from TransformBase. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::ParameterMapType;
  using typename Superclass2::RegistrationType;
  using typename Superclass2::CoordRepType;
  using typename Superclass2::FixedImageType;
  using typename Superclass2::MovingImageType;
  using ITKBaseType = typename Superclass2::ITKBaseType;
  using CombinationTransformType = typename Superclass2::CombinationTransformType;

  /** Other typedef's inherited from Superclass1. */
  using typename Superclass1::IntermediaryDFTransformType;
  using typename Superclass1::VectorImageType;
  using VectorImagePointer = typename VectorImageType::Pointer;

  /** References to the fixed and moving image types. */
  using FixedImageELXType = typename ElastixType::FixedImageType;
  using MovingImageELXType = typename ElastixType::MovingImageType;

  /** Other typedef's.*/
  using DummyImageType = itk::Image<short, Self::SpaceDimension>;
  using DummyIteratorType = itk::ImageRegionConstIterator<DummyImageType>;
  using BSplineTransformPointer = typename BSplineTransformType::Pointer;
  using GenericDeformationFieldRegulizer = typename Superclass1::Superclass;

  /** Typedef's for the diffusion of the deformation field. */
  using VectorReaderType = itk::ImageFileReader<VectorImageType>;
  using VectorType = typename VectorImageType::PixelType;
  using VectorImageIteratorType = itk::ImageRegionIterator<VectorImageType>;
  using GrayValueImageType = FixedImageELXType;
  using GrayValueImagePointer = typename GrayValueImageType::Pointer;
  using GrayValuePixelType = typename GrayValueImageType::PixelType;
  using GrayValueImageIteratorType = itk::ImageRegionIterator<GrayValueImageType>;
  using MaximumImageFilterType = itk::MaximumImageFilter<GrayValueImageType, GrayValueImageType, GrayValueImageType>;
  using DiffusionFilterType = itk::VectorMeanDiffusionImageFilter<VectorImageType, GrayValueImageType>;
  using DiffusionFilterPointer = typename DiffusionFilterType::Pointer;
  using RadiusType = typename VectorImageType::SizeType;
  using ResamplerType1 = itk::ResampleImageFilter<MovingImageELXType, GrayValueImageType, CoordRepType>;
  using ResamplerPointer1 = typename ResamplerType1::Pointer;
  using ResamplerType2 = itk::ResampleImageFilter<GrayValueImageType, GrayValueImageType, CoordRepType>;
  using ResamplerPointer2 = typename ResamplerType2::Pointer;
  using InterpolatorType = itk::BSplineInterpolateImageFunction<GrayValueImageType>;
  using InterpolatorPointer = typename InterpolatorType::Pointer;
  using GrayValueImageReaderType = itk::ImageFileReader<GrayValueImageType>;
  using GrayValueImageReaderPointer = typename GrayValueImageReaderType::Pointer;

  /** Execute stuff before the actual registration:
   * \li Create an initial B-spline grid.
   * \li Create initial registration parameters.
   * \li Setup stuff for the diffusion of the deformation field.
   */
  void
  BeforeRegistration() override;

  /** Execute stuff before each new pyramid resolution:
   * \li upsample the B-spline grid.
   */
  void
  BeforeEachResolution() override;

  /** Execute stuff after each iteration:
   * \li Do a diffusion of the deformation field.
   */
  void
  AfterEachIteration() override;

  /** Execute stuff after registration:
   * \li Destroy things that are not needed anymore in order to free memory.
   */
  void
  AfterRegistration() override;

  /** Set the initial B-spline grid. */
  virtual void
  SetInitialGrid(bool upsampleGridOption);

  /** Upsample the B-spline grid. */
  virtual void
  IncreaseScale();

  /** Function to read transform-parameters from a file. */
  void
  ReadFromFile() override;

  /** Diffuse the deformation field. */
  void
  DiffuseDeformationField();

  /** Method to transform a point.
   * This method just calls the implementation from the
   * GenericDeformationFieldRegulizer. This is necessary, since:
   * The DeformationFieldRegulizerFor is used which expects
   * that its template argument is a BSplineDeformableTransform. This is
   * not the case, because we gave it a BSplineCombinationTransform.
   * This last class has a slightly different behavior of the
   * TransformPoint() method (it does not call the TransformPoint() with
   * with 5 arguments, as the BSplineDeformableTransform does).
   */
  OutputPointType
  TransformPoint(const InputPointType & point) const override;

  /**  Method to transform a point with extra arguments. Just calls
   * the Superclass1's implementation. Has to be present here since it is an
   * overloaded function.
   *
  virtual void TransformPoint(
    const InputPointType & inputPoint,
    OutputPointType & outputPoint,
    WeightsType & weights,
    ParameterIndexArrayType & indices,
    bool & inside ) const;*/

protected:
  /** The constructor. */
  BSplineTransformWithDiffusion();
  /** The destructor. */
  ~BSplineTransformWithDiffusion() override = default;

  /** Member variables. */
  SpacingType m_GridSpacingFactor;

private:
  elxOverrideGetSelfMacro;

  /** Creates a map of the parameters specific for this (derived) transform type. */
  ParameterMapType
  CreateDerivedTransformParametersMap() const override;

  /** Writes its deformation field to a file. */
  void
  WriteDerivedTransformDataToFile() const override;

  /** Member variables for diffusion. */
  DiffusionFilterPointer      m_Diffusion;
  VectorImagePointer          m_DeformationField;
  VectorImagePointer          m_DiffusedField;
  GrayValueImagePointer       m_GrayValueImage1;
  GrayValueImagePointer       m_GrayValueImage2;
  GrayValueImagePointer       m_MovingSegmentationImage;
  GrayValueImagePointer       m_FixedSegmentationImage;
  GrayValueImageReaderPointer m_MovingSegmentationReader;
  GrayValueImageReaderPointer m_FixedSegmentationReader;
  std::string                 m_MovingSegmentationFileName;
  std::string                 m_FixedSegmentationFileName;
  ResamplerPointer1           m_Resampler1;
  ResamplerPointer2           m_Resampler2;
  InterpolatorPointer         m_Interpolator;
  RegionType                  m_DeformationRegion;
  OriginType                  m_DeformationOrigin;
  SpacingType                 m_DeformationSpacing;

  /** Member variables for writing diffusion files. */
  bool               m_WriteDiffusionFiles;
  bool               m_AlsoFixed;
  bool               m_ThresholdBool;
  GrayValuePixelType m_ThresholdHU;
  bool               m_UseMovingSegmentation;
  bool               m_UseFixedSegmentation;

  /** The B-spline parameters, which is going to be filled with zeros. */
  ParametersType m_BSplineParameters;

  /** The internal BSplineTransform, set as a current transform in
   * the combination transform.
   */
  BSplineTransformPointer m_BSplineTransform;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxBSplineTransformWithDiffusion.hxx"
#endif

#endif // end #ifndef elxBSplineTransformWithDiffusion_h
