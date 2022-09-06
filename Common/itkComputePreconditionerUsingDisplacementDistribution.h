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
#ifndef itkComputePreconditionerUsingDisplacementDistribution_h
#define itkComputePreconditionerUsingDisplacementDistribution_h

#include "itkComputeDisplacementDistribution.h"


namespace itk
{
/**\class ComputePreconditionerUsingDisplacementDistribution
 * \brief This is a helper class for the automatic estimation of a preconditioner for the FPSGD optimizer.
 * // update below
 * More specifically this class computes the Jacobian terms related to the automatic
 * parameter estimation for the adaptive stochastic gradient descent optimizer.
 * Details can be found in the TMI paper
 *
 * [1] Y. Qiao, B. van Lew, B.P.F. Lelieveldt, M. Staring
 * Fast Automatic Step Size Estimation for Gradient Descent Optimization of Image Registration
 * IEEE Transactions on Medical Imaging, vol. 35, no. 2, pp. 391 - 403, February 2016
 * http://dx.doi.org/10.1109/TMI.2015.2476354
 */

template <class TFixedImage, class TTransform>
class ITK_TEMPLATE_EXPORT ComputePreconditionerUsingDisplacementDistribution
  : public ComputeDisplacementDistribution<TFixedImage, TTransform>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ComputePreconditionerUsingDisplacementDistribution);

  /** Standard ITK.*/
  using Self = ComputePreconditionerUsingDisplacementDistribution;
  using Superclass = ComputeDisplacementDistribution<TFixedImage, TTransform>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ComputePreconditionerUsingDisplacementDistribution, ComputeDisplacementDistribution);

  /** typedef  */
  using typename Superclass::FixedImageType;
  using typename Superclass::FixedImagePixelType;
  using typename Superclass::TransformType;
  using typename Superclass::TransformPointer;
  using typename Superclass::FixedImageRegionType;
  using typename Superclass::ParametersType;
  using typename Superclass::DerivativeType;
  using typename Superclass::ScalesType;

  using typename Superclass::FixedImageMaskType;
  using typename Superclass::FixedImageMaskPointer;
  using typename Superclass::FixedImageMaskConstPointer;
  using typename Superclass::NonZeroJacobianIndicesType;

  // check
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** Set/get kappa for regularization. */
  itkSetClampMacro(RegularizationKappa, double, 0.0, 1.0);
  itkGetConstReferenceMacro(RegularizationKappa, double);

  /** Set/get maximum step length delta. */
  itkSetMacro(MaximumStepLength, double);
  itkGetConstReferenceMacro(MaximumStepLength, double);

  /** Set/get kappa for condition number. */
  itkSetClampMacro(ConditionNumber, double, 0.0, 10.0);
  itkGetConstReferenceMacro(ConditionNumber, double);

  /** The main function that performs the computation.
   * DO NOT USE.
   */
  void
  Compute(const ParametersType & mu, double & jacg, double & maxJJ, std::string method) override;

  /** The main function that performs the computation.
   * DO NOT USE.
   */
  virtual void
  ComputeDistributionTermsUsingSearchDir(const ParametersType & mu, double & jacg, double & maxJJ, std::string methods);

  /** The main function that performs the computation.
   * B-spline specific thing we tried. Can be removed later.
   */
  void
  ComputeForBSplineOnly(const ParametersType & mu, const double delta, double & maxJJ, ParametersType & preconditioner);

  /** The main function that performs the computation.
   * The aims to be a generic function, working for all transformations.
   */
  virtual void
  Compute(const ParametersType & mu, double & maxJJ, ParametersType & preconditioner);

  virtual void
  ComputeJacobiTypePreconditioner(const ParametersType & mu, double & maxJJ, ParametersType & preconditioner);

  /** Interpolate the preconditioner, for the non-visited entries. */
  virtual void
  PreconditionerInterpolation(ParametersType & preconditioner);

protected:
  ComputePreconditionerUsingDisplacementDistribution();
  ~ComputePreconditionerUsingDisplacementDistribution() override = default;

  using typename Superclass::FixedImageIndexType;
  using typename Superclass::FixedImagePointType;
  using typename Superclass::JacobianType;
  using typename Superclass::JacobianValueType;
  using typename Superclass::ImageSamplerBaseType;
  using typename Superclass::ImageSamplerBasePointer;
  using typename Superclass::ImageFullSamplerType;
  using typename Superclass::ImageFullSamplerPointer;
  using typename Superclass::ImageRandomSamplerBaseType;
  using typename Superclass::ImageRandomSamplerBasePointer;
  using typename Superclass::ImageGridSamplerType;
  using typename Superclass::ImageGridSamplerPointer;
  using typename Superclass::ImageSampleContainerType;
  using typename Superclass::ImageSampleContainerPointer;
  using typename Superclass::TransformJacobianType;
  using typename Superclass::CoordinateRepresentationType;
  using typename Superclass::NumberOfParametersType;

  double m_MaximumStepLength;
  double m_RegularizationKappa;
  double m_ConditionNumber;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkComputePreconditionerUsingDisplacementDistribution.hxx"
#endif

#endif // end #ifndef itkComputePreconditionerUsingDisplacementDistribution_h
