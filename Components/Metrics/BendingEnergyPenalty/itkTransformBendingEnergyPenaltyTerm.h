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
#ifndef itkTransformBendingEnergyPenaltyTerm_h
#define itkTransformBendingEnergyPenaltyTerm_h

#include "itkTransformPenaltyTerm.h"
#include "itkImageGridSampler.h"

namespace itk
{

/**
 * \class TransformBendingEnergyPenaltyTerm
 * \brief A cost function that calculates the bending energy
 * of a transformation.
 *
 * The bending energy is defined as the sum of the spatial
 * second order derivatives of the transformation, as defined in
 * [1]. For rigid and affine transformation this energy is always
 * zero.
 *
 *
 * [1]: D. Rueckert, L. I. Sonoda, C. Hayes, D. L. G. Hill,
 *      M. O. Leach, and D. J. Hawkes, "Nonrigid registration
 *      using free-form deformations: Application to breast MR
 *      images", IEEE Trans. Med. Imaging 18, 712-721, 1999.\n
 * [2]: M. Staring and S. Klein,
 *      "Itk::Transforms supporting spatial derivatives"",
 *      Insight Journal, http://hdl.handle.net/10380/3215.
 *
 * \ingroup Metrics
 */

template <class TFixedImage, class TScalarType>
class ITK_TEMPLATE_EXPORT TransformBendingEnergyPenaltyTerm : public TransformPenaltyTerm<TFixedImage, TScalarType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(TransformBendingEnergyPenaltyTerm);

  /** Standard ITK stuff. */
  using Self = TransformBendingEnergyPenaltyTerm;
  using Superclass = TransformPenaltyTerm<TFixedImage, TScalarType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TransformBendingEnergyPenaltyTerm, TransformPenaltyTerm);

  /** Typedefs inherited from the superclass. */
  using typename Superclass::CoordinateRepresentationType;
  using typename Superclass::MovingImageType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImagePointer;
  using typename Superclass::MovingImageConstPointer;
  using typename Superclass::FixedImageType;
  using typename Superclass::FixedImagePointer;
  using typename Superclass::FixedImageConstPointer;
  using typename Superclass::FixedImageRegionType;
  using typename Superclass::TransformType;
  using typename Superclass::TransformPointer;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::TransformParametersType;
  using typename Superclass::TransformJacobianType;
  using typename Superclass::NumberOfParametersType;
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
  using typename Superclass::ImageSampleContainerType;
  using typename Superclass::ImageSampleContainerPointer;
  using typename Superclass::ScalarType;
  using typename Superclass::ThreaderType;
  using typename Superclass::ThreadInfoType;

  /** Typedef's for the B-spline transform. */
  using typename Superclass::CombinationTransformType;
  using typename Superclass::BSplineOrder1TransformType;
  using typename Superclass::BSplineOrder1TransformPointer;
  using typename Superclass::BSplineOrder2TransformType;
  using typename Superclass::BSplineOrder2TransformPointer;
  using typename Superclass::BSplineOrder3TransformType;
  using typename Superclass::BSplineOrder3TransformPointer;

  /** Typedefs from the AdvancedTransform. */
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::InternalMatrixType;
  using typename Superclass::HessianValueType;
  using typename Superclass::HessianType;

  /** Define the dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** Get the penalty term value. */
  MeasureType
  GetValue(const ParametersType & parameters) const override;

  /** Get the penalty term derivative. */
  void
  GetDerivative(const ParametersType & parameters, DerivativeType & derivative) const override;

  /** Get the penalty term value and derivative. */
  virtual void
  GetValueAndDerivativeSingleThreaded(const ParametersType & parameters,
                                      MeasureType &          value,
                                      DerivativeType &       derivative) const;

  void
  GetValueAndDerivative(const ParametersType & parameters,
                        MeasureType &          value,
                        DerivativeType &       derivative) const override;

  /** Get value and derivatives for each thread. */
  inline void
  ThreadedGetValueAndDerivative(ThreadIdType threadID) override;

  /** Gather the values and derivatives from all threads */
  inline void
  AfterThreadedGetValueAndDerivative(MeasureType & value, DerivativeType & derivative) const override;

  /** Experimental feature: compute SelfHessian */
  void
  GetSelfHessian(const TransformParametersType & parameters, HessianType & H) const override;

  /** Default: 100000 */
  itkSetMacro(NumberOfSamplesForSelfHessian, unsigned int);
  itkGetConstMacro(NumberOfSamplesForSelfHessian, unsigned int);

protected:
  /** Typedefs for indices and points. */
  using typename Superclass::FixedImageIndexType;
  using typename Superclass::FixedImageIndexValueType;
  using typename Superclass::MovingImageIndexType;
  using typename Superclass::FixedImagePointType;
  using typename Superclass::MovingImagePointType;
  using typename Superclass::MovingImageContinuousIndexType;
  using typename Superclass::NonZeroJacobianIndicesType;

  /** Typedefs for SelfHessian */
  using SelfHessianSamplerType = ImageGridSampler<FixedImageType>;

  /** The constructor. */
  TransformBendingEnergyPenaltyTerm();

  /** The destructor. */
  ~TransformBendingEnergyPenaltyTerm() override = default;

private:
  unsigned int m_NumberOfSamplesForSelfHessian;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkTransformBendingEnergyPenaltyTerm.hxx"
#endif

#endif // #ifndef itkTransformBendingEnergyPenaltyTerm_h
