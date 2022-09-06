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
#ifndef itkPCAMetric_F_multithreaded_h
#define itkPCAMetric_F_multithreaded_h

#include "itkAdvancedImageToImageMetric.h"

#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkImageRandomCoordinateSampler.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkExtractImageFilter.h"
#include <vector>

namespace itk
{
template <class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT PCAMetric : public AdvancedImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(PCAMetric);

  /** Standard class typedefs. */
  using Self = PCAMetric;
  using Superclass = AdvancedImageToImageMetric<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  using typename Superclass::FixedImageRegionType;
  using FixedImageSizeType = typename FixedImageRegionType::SizeType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(PCAMetric, AdvancedImageToImageMetric);

  /** Set functions. */
  itkSetMacro(SubtractMean, bool);
  itkSetMacro(GridSize, FixedImageSizeType);
  itkSetMacro(TransformIsStackTransform, bool);
  itkSetMacro(NumEigenValues, unsigned int);

  /** Typedefs from the superclass. */
  using typename Superclass::CoordinateRepresentationType;
  using typename Superclass::MovingImageType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImageConstPointer;
  using typename Superclass::FixedImageType;
  using typename Superclass::FixedImageConstPointer;
  using typename Superclass::TransformType;
  using typename Superclass::TransformPointer;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::TransformParametersType;
  using typename Superclass::TransformJacobianType;
  using typename Superclass::InterpolatorType;
  using typename Superclass::InterpolatorPointer;
  using typename Superclass::RealType;
  using typename Superclass::GradientPixelType;
  using typename Superclass::GradientImageType;
  using typename Superclass::GradientImagePointer;
  using typename Superclass::GradientImageFilterType;
  using typename Superclass::GradientImageFilterPointer;
  using typename Superclass::FixedImageMaskType;
  using typename Superclass::FixedImageMaskPointer;
  using typename Superclass::MovingImageMaskType;
  using typename Superclass::MovingImageMaskPointer;
  using typename Superclass::MeasureType;
  using typename Superclass::DerivativeType;
  using typename Superclass::ParametersType;
  using typename Superclass::FixedImagePixelType;
  using typename Superclass::MovingImageRegionType;
  using typename Superclass::ImageSamplerType;
  using typename Superclass::ImageSamplerPointer;
  using typename Superclass::ImageSampleContainerType;
  using typename Superclass::ImageSampleContainerPointer;
  using typename Superclass::FixedImageLimiterType;
  using typename Superclass::MovingImageLimiterType;
  using typename Superclass::FixedImageLimiterOutputType;
  using typename Superclass::MovingImageLimiterOutputType;
  using typename Superclass::MovingImageDerivativeScalesType;
  using DerivativeValueType = typename DerivativeType::ValueType;
  using typename Superclass::ThreaderType;
  using typename Superclass::ThreadInfoType;

  using MatrixType = vnl_matrix<RealType>;
  using DerivativeMatrixType = vnl_matrix<DerivativeValueType>;

  //    using MatrixType = vnl_matrix< double >;
  //    using DerivativeMatrixType = vnl_matrix< double >;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Get the value for single valued optimizers. */
  MeasureType
  GetValue(const TransformParametersType & parameters) const override;

  /** Get the derivatives of the match measure. */
  void
  GetDerivative(const TransformParametersType & parameters, DerivativeType & derivative) const override;

  /** Get value and derivatives for multiple valued optimizers. */
  void
  GetValueAndDerivativeSingleThreaded(const TransformParametersType & parameters,
                                      MeasureType &                   Value,
                                      DerivativeType &                Derivative) const;

  void
  GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType &                   Value,
                        DerivativeType &                Derivative) const override;

  /** Initialize the Metric by making sure that all the components
   *  are present and plugged together correctly.
   * \li Call the superclass' implementation.   */

  void
  Initialize() override;

protected:
  PCAMetric();
  ~PCAMetric() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Protected Typedefs ******************/

  /** Typedefs inherited from superclass */
  using typename Superclass::FixedImageIndexType;
  using typename Superclass::FixedImageIndexValueType;
  using typename Superclass::MovingImageIndexType;
  using typename Superclass::FixedImagePointType;
  using FixedImageContinuousIndexType =
    typename itk::ContinuousIndex<CoordinateRepresentationType, FixedImageDimension>;
  using typename Superclass::MovingImagePointType;
  using typename Superclass::MovingImageContinuousIndexType;
  using typename Superclass::BSplineInterpolatorType;
  using typename Superclass::CentralDifferenceGradientFilterType;
  using typename Superclass::MovingImageDerivativeType;
  using typename Superclass::NonZeroJacobianIndicesType;

  /** Computes the innerproduct of transform Jacobian with moving image gradient.
   * The results are stored in imageJacobian, which is supposed
   * to have the right size (same length as Jacobian's number of columns). */
  void
  EvaluateTransformJacobianInnerProduct(const TransformJacobianType &     jacobian,
                                        const MovingImageDerivativeType & movingImageDerivative,
                                        DerivativeType &                  imageJacobian) const override;

  /** Get value and derivatives for each thread. */
  inline void
  ThreadedGetSamples(ThreadIdType threadID);

  inline void
  ThreadedComputeDerivative(ThreadIdType threadID);

  /** Gather the values and derivatives from all threads */
  inline void
  AfterThreadedGetSamples(MeasureType & value) const;

  inline void
  AfterThreadedComputeDerivative(DerivativeType & derivative) const;

  /** Helper function to launch the threads. */
  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  GetSamplesThreaderCallback(void * arg);

  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  ComputeDerivativeThreaderCallback(void * arg);

  /** Helper functions to launch the threads. */
  void
  LaunchGetSamplesThreaderCallback() const;

  void
  LaunchComputeDerivativeThreaderCallback() const;

  /** Initialize some multi-threading related parameters. */
  void
  InitializeThreadingParameters() const override;

private:
  struct PCAMetricMultiThreaderParameterType
  {
    Self * m_Metric;
  };

  PCAMetricMultiThreaderParameterType m_PCAMetricThreaderParameters;

  struct PCAMetricGetSamplesPerThreadStruct
  {
    SizeValueType                    st_NumberOfPixelsCounted;
    MatrixType                       st_DataBlock;
    std::vector<FixedImagePointType> st_ApprovedSamples;
    DerivativeType                   st_Derivative;
  };

  itkPadStruct(ITK_CACHE_LINE_ALIGNMENT, PCAMetricGetSamplesPerThreadStruct, PaddedPCAMetricGetSamplesPerThreadStruct);

  itkAlignedTypedef(ITK_CACHE_LINE_ALIGNMENT,
                    PaddedPCAMetricGetSamplesPerThreadStruct,
                    AlignedPCAMetricGetSamplesPerThreadStruct);

  mutable std::vector<AlignedPCAMetricGetSamplesPerThreadStruct> m_PCAMetricGetSamplesPerThreadVariables;

  unsigned int m_G;
  unsigned int m_LastDimIndex;

  /** Bool to determine if we want to subtract the mean derivate from the derivative elements. */
  bool m_SubtractMean{ false };

  /** GridSize of B-spline transform. */
  FixedImageSizeType m_GridSize;

  /** Bool to indicate if the transform used is a stacktransform. Set by elx files. */
  bool m_TransformIsStackTransform{ false };

  /** Integer to indicate how many eigenvalues you want to use in the metric */
  unsigned int m_NumEigenValues{ 6 };

  /** Matrices, needed for derivative calculation */
  mutable std::vector<unsigned int> m_PixelStartIndex;
  mutable MatrixType                m_Atmm;
  mutable DerivativeMatrixType      m_vSAtmm;
  mutable DerivativeMatrixType      m_CSv;
  mutable DerivativeMatrixType      m_Sv;
  mutable DerivativeMatrixType      m_vdSdmu_part1;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkPCAMetric_F_multithreaded.hxx"
#endif

#endif // end #ifndef itkPCAMetric_F_multithreaded_h
