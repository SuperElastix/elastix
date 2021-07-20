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

namespace itk
{
template <class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT PCAMetric : public AdvancedImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  /** Standard class typedefs. */
  typedef PCAMetric                                             Self;
  typedef AdvancedImageToImageMetric<TFixedImage, TMovingImage> Superclass;
  typedef SmartPointer<Self>                                    Pointer;
  typedef SmartPointer<const Self>                              ConstPointer;

  typedef typename Superclass::FixedImageRegionType FixedImageRegionType;
  typedef typename FixedImageRegionType::SizeType   FixedImageSizeType;

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
  typedef typename Superclass::CoordinateRepresentationType    CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType                 MovingImageType;
  typedef typename Superclass::MovingImagePixelType            MovingImagePixelType;
  typedef typename Superclass::MovingImageConstPointer         MovingImageConstPointer;
  typedef typename Superclass::FixedImageType                  FixedImageType;
  typedef typename Superclass::FixedImageConstPointer          FixedImageConstPointer;
  typedef typename Superclass::TransformType                   TransformType;
  typedef typename Superclass::TransformPointer                TransformPointer;
  typedef typename Superclass::InputPointType                  InputPointType;
  typedef typename Superclass::OutputPointType                 OutputPointType;
  typedef typename Superclass::TransformParametersType         TransformParametersType;
  typedef typename Superclass::TransformJacobianType           TransformJacobianType;
  typedef typename Superclass::InterpolatorType                InterpolatorType;
  typedef typename Superclass::InterpolatorPointer             InterpolatorPointer;
  typedef typename Superclass::RealType                        RealType;
  typedef typename Superclass::GradientPixelType               GradientPixelType;
  typedef typename Superclass::GradientImageType               GradientImageType;
  typedef typename Superclass::GradientImagePointer            GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType         GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer      GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType              FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer           FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType             MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer          MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                     MeasureType;
  typedef typename Superclass::DerivativeType                  DerivativeType;
  typedef typename Superclass::ParametersType                  ParametersType;
  typedef typename Superclass::FixedImagePixelType             FixedImagePixelType;
  typedef typename Superclass::MovingImageRegionType           MovingImageRegionType;
  typedef typename Superclass::ImageSamplerType                ImageSamplerType;
  typedef typename Superclass::ImageSamplerPointer             ImageSamplerPointer;
  typedef typename Superclass::ImageSampleContainerType        ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer     ImageSampleContainerPointer;
  typedef typename Superclass::FixedImageLimiterType           FixedImageLimiterType;
  typedef typename Superclass::MovingImageLimiterType          MovingImageLimiterType;
  typedef typename Superclass::FixedImageLimiterOutputType     FixedImageLimiterOutputType;
  typedef typename Superclass::MovingImageLimiterOutputType    MovingImageLimiterOutputType;
  typedef typename Superclass::MovingImageDerivativeScalesType MovingImageDerivativeScalesType;
  typedef typename DerivativeType::ValueType                   DerivativeValueType;
  typedef typename Superclass::ThreaderType                    ThreaderType;
  typedef typename Superclass::ThreadInfoType                  ThreadInfoType;

  typedef vnl_matrix<RealType>            MatrixType;
  typedef vnl_matrix<DerivativeValueType> DerivativeMatrixType;

  //    typedef vnl_matrix< double > MatrixType;
  //    typedef vnl_matrix< double > DerivativeMatrixType;

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
  Initialize(void) override;

protected:
  PCAMetric();
  ~PCAMetric() override;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Protected Typedefs ******************/

  /** Typedefs inherited from superclass */
  typedef typename Superclass::FixedImageIndexType      FixedImageIndexType;
  typedef typename Superclass::FixedImageIndexValueType FixedImageIndexValueType;
  typedef typename Superclass::MovingImageIndexType     MovingImageIndexType;
  typedef typename Superclass::FixedImagePointType      FixedImagePointType;
  typedef typename itk::ContinuousIndex<CoordinateRepresentationType, FixedImageDimension>
                                                                   FixedImageContinuousIndexType;
  typedef typename Superclass::MovingImagePointType                MovingImagePointType;
  typedef typename Superclass::MovingImageContinuousIndexType      MovingImageContinuousIndexType;
  typedef typename Superclass::BSplineInterpolatorType             BSplineInterpolatorType;
  typedef typename Superclass::CentralDifferenceGradientFilterType CentralDifferenceGradientFilterType;
  typedef typename Superclass::MovingImageDerivativeType           MovingImageDerivativeType;
  typedef typename Superclass::NonZeroJacobianIndicesType          NonZeroJacobianIndicesType;

  /** Computes the innerproduct of transform Jacobian with moving image gradient.
   * The results are stored in imageJacobian, which is supposed
   * to have the right size (same length as Jacobian's number of columns). */
  void
  EvaluateTransformJacobianInnerProduct(const TransformJacobianType &     jacobian,
                                        const MovingImageDerivativeType & movingImageDerivative,
                                        DerivativeType &                  imageJacobian) const override;

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

  mutable AlignedPCAMetricGetSamplesPerThreadStruct * m_PCAMetricGetSamplesPerThreadVariables;
  mutable ThreadIdType                                m_PCAMetricGetSamplesPerThreadVariablesSize;

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
  LaunchGetSamplesThreaderCallback(void) const;

  void
  LaunchComputeDerivativeThreaderCallback(void) const;

  /** Initialize some multi-threading related parameters. */
  void
  InitializeThreadingParameters(void) const override;

private:
  PCAMetric(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  unsigned int m_G;
  unsigned int m_LastDimIndex;

  /** Bool to determine if we want to subtract the mean derivate from the derivative elements. */
  bool m_SubtractMean;

  /** GridSize of B-spline transform. */
  FixedImageSizeType m_GridSize;

  /** Bool to indicate if the transform used is a stacktransform. Set by elx files. */
  bool m_TransformIsStackTransform;

  /** Integer to indicate how many eigenvalues you want to use in the metric */
  unsigned int m_NumEigenValues;

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
