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
#ifndef itkParzenWindowHistogramImageToImageMetric_h
#define itkParzenWindowHistogramImageToImageMetric_h

#include "itkAdvancedImageToImageMetric.h"
#include "itkKernelFunctionBase2.h"
#include <vector>


namespace itk
{
/**
 * \class ParzenWindowHistogramImageToImageMetric
 * \brief A base class for image metrics based on a joint histogram
 * computed using Parzen Windowing
 *
 * The calculations are based on the method of Mattes/Thevenaz/Unser [1,2,3]
 * where the probability density distribution are estimated using
 * Parzen histograms.
 *
 * Once the PDF's have been constructed, the metric value and derivative
 * can be computed. Inheriting classes should make sure to call
 * the function ComputePDFs(AndPDFDerivatives) before using m_JointPDF and m_Alpha
 * (and m_JointPDFDerivatives).
 *
 * This class does not define the GetValue/GetValueAndDerivative methods.
 * This is the task of inheriting classes.
 *
 * The code is based on the itk::MattesMutualInformationImageToImageMetric,
 * but largely rewritten. Some important features:
 *  - It inherits from AdvancedImageToImageMetric, which provides a lot of
 *    general functionality.
 *  - It splits up some functions in subfunctions.
 *  - The Parzen window order can be chosen.
 *  - A fixed and moving number of histogram bins can be chosen.
 *  - More use of iterators instead of raw buffer pointers.
 *  - An optional FiniteDifference derivative estimation.
 *
 * \warning This class is not thread safe due the member data structures
 *  used to the store the sampled points and the marginal and joint pdfs.
 *
 * References:\n
 * [1] "Nonrigid multimodality image registration"\n
 *      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank\n
 *      Medical Imaging 2001: Image Processing, 2001, pp. 1609-1620.\n
 * [2] "PET-CT Image Registration in the Chest Using Free-form Deformations"\n
 *      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank\n
 *      IEEE Transactions in Medical Imaging. To Appear.\n
 * [3] "Optimization of Mutual Information for MultiResolution Image
 *      Registration"\n
 *      P. Thevenaz and M. Unser\n
 *      IEEE Transactions in Image Processing, 9(12) December 2000.\n
 *
 *
 * \ingroup Metrics
 */

template <class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT ParzenWindowHistogramImageToImageMetric
  : public AdvancedImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ParzenWindowHistogramImageToImageMetric);

  /** Standard class typedefs. */
  using Self = ParzenWindowHistogramImageToImageMetric;
  using Superclass = AdvancedImageToImageMetric<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ParzenWindowHistogramImageToImageMetric, AdvancedImageToImageMetric);

  /** Typedefs from the superclass. */
  using typename Superclass::CoordinateRepresentationType;
  using typename Superclass::MovingImageType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImageConstPointer;
  using typename Superclass::FixedImageType;
  using typename Superclass::FixedImageConstPointer;
  using typename Superclass::FixedImageRegionType;
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
  using typename Superclass::DerivativeValueType;
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
  using typename Superclass::ThreaderType;
  using typename Superclass::ThreadInfoType;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Initialize the Metric by
   * (1) Call the superclass' implementation
   * (2) InitializeHistograms()
   * (3) InitializeKernels()
   * (4) Resize AlphaDerivatives
   */
  void
  Initialize() override;

  /** Get the derivatives of the match measure. This method simply calls the
   * the GetValueAndDerivative, since this will be mostly almost as fast
   * as just computing the derivative.
   */
  void
  GetDerivative(const ParametersType & parameters, DerivativeType & Derivative) const override;

  /**  Get the value and derivatives for single valued optimizers.
   * This method calls this->GetValueAndAnalyticDerivative or
   * this->GetValueAndFiniteDifferenceDerivative, depending on the bool
   * m_UseFiniteDifferenceDerivative.
   */
  void
  GetValueAndDerivative(const ParametersType & parameters,
                        MeasureType &          value,
                        DerivativeType &       derivative) const override;

  /** Number of bins to use for the fixed image in the histogram.
   * Typical value is 32.  The minimum value is 4 due to the padding
   * required by the Parzen windowing with a cubic B-spline kernel. Note
   * that even if the metric is used on binary images, the number of bins
   * should at least be equal to four.
   */
  itkSetClampMacro(NumberOfFixedHistogramBins, unsigned long, 4, NumericTraits<unsigned long>::max());
  itkGetConstMacro(NumberOfFixedHistogramBins, unsigned long);

  /** Number of bins to use for the moving image in the histogram.
   * Typical value is 32.  The minimum value is 4 due to the padding
   * required by the Parzen windowing with a cubic B-spline kernel. Note
   * that even if the metric is used on binary images, the number of bins
   * should at least be equal to four.
   */
  itkSetClampMacro(NumberOfMovingHistogramBins, unsigned long, 4, NumericTraits<unsigned long>::max());
  itkGetConstMacro(NumberOfMovingHistogramBins, unsigned long);

  /** The B-spline order of the fixed Parzen window; default: 0 */
  itkSetClampMacro(FixedKernelBSplineOrder, unsigned int, 0, 3);
  itkGetConstMacro(FixedKernelBSplineOrder, unsigned int);

  /** The B-spline order of the moving B-spline order; default: 3 */
  itkSetClampMacro(MovingKernelBSplineOrder, unsigned int, 0, 3);
  itkGetConstMacro(MovingKernelBSplineOrder, unsigned int);

  /** Option to use explicit PDF derivatives, which requires a lot
   * of memory in case of many parameters.
   */
  itkSetMacro(UseExplicitPDFDerivatives, bool);
  itkGetConstReferenceMacro(UseExplicitPDFDerivatives, bool);
  itkBooleanMacro(UseExplicitPDFDerivatives);

  /** Whether you plan to call the GetDerivative/GetValueAndDerivative method or not.
   * This option should be set before calling Initialize(); Default: false.
   */
  itkSetMacro(UseDerivative, bool);
  itkGetConstMacro(UseDerivative, bool);

  /** Whether you want to use a finite difference implementation of the metric's derivative.
   * This option should be set before calling Initialize(); Default: false.
   */
  itkSetMacro(UseFiniteDifferenceDerivative, bool);
  itkGetConstMacro(UseFiniteDifferenceDerivative, bool);

  /** For computing the finite difference derivative, the perturbation (delta) of the
   * transform parameters; default: 1.0.
   * mu_right= mu + delta*e_k
   */
  itkSetMacro(FiniteDifferencePerturbation, double);
  itkGetConstMacro(FiniteDifferencePerturbation, double);

protected:
  /** The constructor. */
  ParzenWindowHistogramImageToImageMetric();

  /** The destructor. */
  ~ParzenWindowHistogramImageToImageMetric() override = default;

  /** Print Self. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Protected Typedefs ******************/

  /** Typedefs inherited from superclass. */
  using typename Superclass::FixedImageIndexType;
  using typename Superclass::FixedImageIndexValueType;
  using OffsetValueType = typename FixedImageType::OffsetValueType;
  using typename Superclass::MovingImageIndexType;
  using typename Superclass::FixedImagePointType;
  using typename Superclass::MovingImagePointType;
  using typename Superclass::MovingImageContinuousIndexType;
  using typename Superclass::BSplineInterpolatorType;
  using typename Superclass::MovingImageDerivativeType;
  using typename Superclass::CentralDifferenceGradientFilterType;
  using typename Superclass::NonZeroJacobianIndicesType;

  /** Typedefs for the PDFs and PDF derivatives. */
  using PDFValueType = double;
  using PDFDerivativeValueType = float;
  using MarginalPDFType = Array<PDFValueType>;
  using JointPDFType = Image<PDFValueType, 2>;
  using JointPDFPointer = typename JointPDFType::Pointer;
  using JointPDFDerivativesType = Image<PDFDerivativeValueType, 3>;
  using JointPDFDerivativesPointer = typename JointPDFDerivativesType::Pointer;
  using IncrementalMarginalPDFType = Image<PDFValueType, 2>;
  using IncrementalMarginalPDFPointer = typename IncrementalMarginalPDFType::Pointer;
  using JointPDFIndexType = JointPDFType::IndexType;
  using JointPDFRegionType = JointPDFType::RegionType;
  using JointPDFSizeType = JointPDFType::SizeType;
  using JointPDFDerivativesIndexType = JointPDFDerivativesType::IndexType;
  using JointPDFDerivativesRegionType = JointPDFDerivativesType::RegionType;
  using JointPDFDerivativesSizeType = JointPDFDerivativesType::SizeType;
  using IncrementalMarginalPDFIndexType = IncrementalMarginalPDFType::IndexType;
  using IncrementalMarginalPDFRegionType = IncrementalMarginalPDFType::RegionType;
  using IncrementalMarginalPDFSizeType = IncrementalMarginalPDFType::SizeType;
  using ParzenValueContainerType = Array<PDFValueType>;

  /** Typedefs for Parzen kernel. */
  using KernelFunctionType = KernelFunctionBase2<PDFValueType>;
  using KernelFunctionPointer = typename KernelFunctionType::Pointer;

  /** Protected variables **************************** */

  /** Variables for Alpha (the normalization factor of the histogram). */
  mutable double         m_Alpha;
  mutable DerivativeType m_PerturbedAlphaRight;
  mutable DerivativeType m_PerturbedAlphaLeft;

  /** Variables for the pdfs (actually: histograms). */
  mutable MarginalPDFType       m_FixedImageMarginalPDF;
  mutable MarginalPDFType       m_MovingImageMarginalPDF;
  JointPDFPointer               m_JointPDF;
  JointPDFDerivativesPointer    m_JointPDFDerivatives;
  JointPDFDerivativesPointer    m_IncrementalJointPDFRight;
  JointPDFDerivativesPointer    m_IncrementalJointPDFLeft;
  IncrementalMarginalPDFPointer m_FixedIncrementalMarginalPDFRight;
  IncrementalMarginalPDFPointer m_MovingIncrementalMarginalPDFRight;
  IncrementalMarginalPDFPointer m_FixedIncrementalMarginalPDFLeft;
  IncrementalMarginalPDFPointer m_MovingIncrementalMarginalPDFLeft;
  mutable JointPDFRegionType    m_JointPDFWindow; // no need for mutable anymore?
  double                        m_MovingImageNormalizedMin;
  double                        m_FixedImageNormalizedMin;
  double                        m_FixedImageBinSize;
  double                        m_MovingImageBinSize;
  double                        m_FixedParzenTermToIndexOffset;
  double                        m_MovingParzenTermToIndexOffset;

  /** Kernels for computing Parzen histograms and derivatives. */
  KernelFunctionPointer m_FixedKernel;
  KernelFunctionPointer m_MovingKernel;
  KernelFunctionPointer m_DerivativeMovingKernel;

  /** Initialize threading related parameters. */
  void
  InitializeThreadingParameters() const override;

  /** Multi-threaded versions of the ComputePDF function. */
  inline void
  ThreadedComputePDFs(ThreadIdType threadId);

  /** Single-threadedly accumulate results. */
  inline void
  AfterThreadedComputePDFs() const;

  /** Helper function to launch the threads. */
  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  ComputePDFsThreaderCallback(void * arg);

  /** Helper function to launch the threads. */
  void
  LaunchComputePDFsThreaderCallback() const;

  /** Compute the Parzen values given an image value and a starting histogram index
   * Compute the values at (parzenWindowIndex - parzenWindowTerm + k) for
   * k = 0 ... kernelsize-1
   * Returns the values in a ParzenValueContainer, which is supposed to have
   * the right size already.
   */
  void
  EvaluateParzenValues(double                     parzenWindowTerm,
                       OffsetValueType            parzenWindowIndex,
                       const KernelFunctionType * kernel,
                       ParzenValueContainerType & parzenValues) const;

  /** Update the joint PDF with a pixel pair; on demand also updates the
   * pdf derivatives (if the Jacobian pointers are nonzero).
   */
  virtual void
  UpdateJointPDFAndDerivatives(const RealType &                   fixedImageValue,
                               const RealType &                   movingImageValue,
                               const DerivativeType *             imageJacobian,
                               const NonZeroJacobianIndicesType * nzji,
                               JointPDFType *                     jointPDF) const;

  /** Update the joint PDF and the incremental pdfs.
   * The input is a pixel pair (fixed, moving, moving mask) and
   * a set of moving image/mask values when using mu+delta*e_k, for
   * each k that has a nonzero Jacobian. And for mu-delta*e_k of course.
   * Also updates the PerturbedAlpha's
   * This function is used when UseFiniteDifferenceDerivative is true.
   *
   * \todo The IsInsideMovingMask return bools are converted to doubles (1 or 0) to
   * simplify the computation. But this may not be necessary.
   */
  virtual void
  UpdateJointPDFAndIncrementalPDFs(RealType                           fixedImageValue,
                                   RealType                           movingImageValue,
                                   RealType                           movingMaskValue,
                                   const DerivativeType &             movingImageValuesRight,
                                   const DerivativeType &             movingImageValuesLeft,
                                   const DerivativeType &             movingMaskValuesRight,
                                   const DerivativeType &             movingMaskValuesLeft,
                                   const NonZeroJacobianIndicesType & nzji) const;

  /** Update the pdf derivatives
   * adds -image_jac[mu]*factor to the bin
   * with index [ mu, pdfIndex[0], pdfIndex[1] ] for all mu.
   * This function should only be called from UpdateJointPDFAndDerivatives.
   */
  void
  UpdateJointPDFDerivatives(const JointPDFIndexType &          pdfIndex,
                            double                             factor,
                            const DerivativeType &             imageJacobian,
                            const NonZeroJacobianIndicesType & nzji) const;

  /** Multiply the pdf entries by the given normalization factor. */
  void
  NormalizeJointPDF(JointPDFType * pdf, const double factor) const;

  /** Multiply the pdf derivatives entries by the given normalization factor. */
  void
  NormalizeJointPDFDerivatives(JointPDFDerivativesType * pdf, const double factor) const;

  /** Compute marginal pdfs by summing over the joint pdf
   * direction = 0: fixed marginal pdf
   * direction = 1: moving marginal pdf
   */
  void
  ComputeMarginalPDF(const JointPDFType * jointPDF, MarginalPDFType & marginalPDF, const unsigned int direction) const;

  /** Compute incremental marginal pdfs. Integrates the incremental PDF
   * to obtain the fixed and moving marginal pdfs at once.
   */
  virtual void
  ComputeIncrementalMarginalPDFs(const JointPDFDerivativesType * incrementalPDF,
                                 IncrementalMarginalPDFType *    fixedIncrementalMarginalPDF,
                                 IncrementalMarginalPDFType *    movingIncrementalMarginalPDF) const;

  /** Compute PDFs and pdf derivatives; Loops over the fixed image samples and constructs
   * the m_JointPDF, m_JointPDFDerivatives, and m_Alpha.
   * The JointPDF and Alpha and its derivatives are related as follows:
   * p = m_Alpha * m_JointPDF
   * dp/dmu = m_Alpha * m_JointPDFDerivatives
   * So, the JointPDF is more like a histogram than a true pdf...
   * The histograms are left unnormalized since it may be faster to
   * not do this explicitly.
   */
  virtual void
  ComputePDFsAndPDFDerivatives(const ParametersType & parameters) const;

  /** Compute PDFs and incremental pdfs (which you can use to compute finite
   * difference estimate of the derivative).
   * Loops over the fixed image samples and constructs the m_JointPDF,
   * m_IncrementalJointPDF<Right/Left>, m_Alpha, and m_PerturbedAlpha<Right/Left>.
   *
   * mu = input parameters vector
   * jh(mu) = m_JointPDF(:,:) = joint histogram
   * ihr(k) = m_IncrementalJointPDFRight(k,:,:)
   * ihl(k) = m_IncrementalJointPDFLeft(k,:,:)
   * a(mu) = m_Alpha
   * par(k) = m_PerturbedAlphaRight(k)
   * pal(k) = m_PerturbedAlphaLeft(k)
   * size(ihr) = = size(ihl) = nrofparams * nrofmovingbins * nroffixedbins
   *
   * ihr and ihl are determined such that:
   * jh(mu+delta*e_k) = jh(mu) + ihr(k)
   * jh(mu-delta*e_k) = jh(mu) + ihl(k)
   * where e_k is the unit vector.
   *
   * the pdf can be derived with:
   * p(mu+delta*e_k) = ( par(k) ) * jh(mu+delta*e_k)
   * p(mu-delta*e_k) = ( pal(k) ) * jh(mu-delta*e_k)
   */
  virtual void
  ComputePDFsAndIncrementalPDFs(const ParametersType & parameters) const;

  /** Compute PDFs; Loops over the fixed image samples and constructs
   * the m_JointPDF and m_Alpha
   * The JointPDF and Alpha are related as follows:
   * p = m_Alpha * m_JointPDF
   * So, the JointPDF is more like a histogram than a true pdf...
   * The histogram is left unnormalised since it may be faster to
   * not do this explicitly.
   */
  virtual void
  ComputePDFsSingleThreaded(const ParametersType & parameters) const;

  virtual void
  ComputePDFs(const ParametersType & parameters) const;

  /** Some initialization functions, called by Initialize. */
  virtual void
  InitializeHistograms();

  virtual void
  InitializeKernels();

  /** Get the value and analytic derivatives for single valued optimizers.
   * Called by GetValueAndDerivative if UseFiniteDifferenceDerivative == false
   * Implement this method in subclasses.
   */
  virtual void
  GetValueAndAnalyticDerivative(const ParametersType & itkNotUsed(parameters),
                                MeasureType &          itkNotUsed(value),
                                DerivativeType &       itkNotUsed(derivative)) const
  {}

  /** Get the value and finite difference derivatives for single valued optimizers.
   * Called by GetValueAndDerivative if UseFiniteDifferenceDerivative == true
   * Implement this method in subclasses.
   */
  virtual void
  GetValueAndFiniteDifferenceDerivative(const ParametersType & itkNotUsed(parameters),
                                        MeasureType &          itkNotUsed(value),
                                        DerivativeType &       itkNotUsed(derivative)) const
  {}

private:
  /** Threading related parameters. */
  mutable std::vector<JointPDFPointer> m_ThreaderJointPDFs;

  /** Helper structs that multi-threads the computation of
   * the metric derivative using ITK threads.
   */
  struct ParzenWindowHistogramMultiThreaderParameterType // can't we use the one from AdvancedImageToImageMetric ?
  {
    Self * m_Metric;
  };
  ParzenWindowHistogramMultiThreaderParameterType m_ParzenWindowHistogramThreaderParameters;

  struct ParzenWindowHistogramGetValueAndDerivativePerThreadStruct
  {
    SizeValueType   st_NumberOfPixelsCounted;
    JointPDFPointer st_JointPDF;
  };
  itkPadStruct(ITK_CACHE_LINE_ALIGNMENT,
               ParzenWindowHistogramGetValueAndDerivativePerThreadStruct,
               PaddedParzenWindowHistogramGetValueAndDerivativePerThreadStruct);
  itkAlignedTypedef(ITK_CACHE_LINE_ALIGNMENT,
                    PaddedParzenWindowHistogramGetValueAndDerivativePerThreadStruct,
                    AlignedParzenWindowHistogramGetValueAndDerivativePerThreadStruct);
  mutable std::vector<AlignedParzenWindowHistogramGetValueAndDerivativePerThreadStruct>
    m_ParzenWindowHistogramGetValueAndDerivativePerThreadVariables;

  /** Variables that can/should be accessed by their Set/Get functions. */
  unsigned long m_NumberOfFixedHistogramBins;
  unsigned long m_NumberOfMovingHistogramBins;
  unsigned int  m_FixedKernelBSplineOrder;
  unsigned int  m_MovingKernelBSplineOrder;
  bool          m_UseDerivative;
  bool          m_UseExplicitPDFDerivatives;
  bool          m_UseFiniteDifferenceDerivative;
  double        m_FiniteDifferencePerturbation;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkParzenWindowHistogramImageToImageMetric.hxx"
#endif

#endif // end #ifndef itkParzenWindowHistogramImageToImageMetric_h
