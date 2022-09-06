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
#ifndef itkWeightedCombinationTransform_h
#define itkWeightedCombinationTransform_h

#include "itkAdvancedTransform.h"

namespace itk
{

/** \class WeightedCombinationTransform
 * \brief Implements a weighted linear combination of multiple transforms.
 *
 * This transform implements:
 * \f[T(x) = x + \sum_i w_i ( T_i(x) - x )\f]
 * where \f$w_i\f$ are the weights, which are the transform's parameters, and
 * can be set/get by Set/GetParameters().
 *
 * Alternatively, if the NormalizeWeights parameter is set to true,
 * the transformation is as follows:
 * \f[T(x) = \sum_i w_i T_i(x) / \sum_i w_i\f]
 *
 * \ingroup Transforms
 *
 */
template <class TScalarType, unsigned int NInputDimensions = 3, unsigned int NOutputDimensions = 3>
class ITK_TEMPLATE_EXPORT WeightedCombinationTransform
  : public AdvancedTransform<TScalarType, NInputDimensions, NOutputDimensions>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(WeightedCombinationTransform);

  /** Standard class typedefs. */
  using Self = WeightedCombinationTransform;
  using Superclass = AdvancedTransform<TScalarType, NInputDimensions, NOutputDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New method for creating an object using a factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(WeightedCombinationTransform, AdvancedTransform);

  /** Dimension of the domain space. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, NInputDimensions);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, NOutputDimensions);

  /** Typedefs from the Superclass. */
  using typename Superclass::ScalarType;
  using typename Superclass::ParametersType;
  using typename Superclass::NumberOfParametersType;
  using typename Superclass::JacobianType;
  using typename Superclass::InputVectorType;
  using typename Superclass::OutputVectorType;
  using typename Superclass::InputCovariantVectorType;
  using typename Superclass::OutputCovariantVectorType;
  using typename Superclass::InputVnlVectorType;
  using typename Superclass::OutputVnlVectorType;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::NonZeroJacobianIndicesType;
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;

  /** New typedefs in this class: */
  using TransformType = Transform<TScalarType, NInputDimensions, NOutputDimensions>;
  /** \todo: shouldn't these be ConstPointers? */
  using TransformPointer = typename TransformType::Pointer;
  using TransformContainerType = std::vector<TransformPointer>;

  /**  Method to transform a point. */
  OutputPointType
  TransformPoint(const InputPointType & inputPoint) const override;

  /** These vector transforms are not implemented for this transform. */
  OutputVectorType
  TransformVector(const InputVectorType &) const override
  {
    itkExceptionMacro(
      << "TransformVector(const InputVectorType &) is not implemented for WeightedCombinationTransform");
  }


  OutputVnlVectorType
  TransformVector(const InputVnlVectorType &) const override
  {
    itkExceptionMacro(
      << "TransformVector(const InputVnlVectorType &) is not implemented for WeightedCombinationTransform");
  }


  OutputCovariantVectorType
  TransformCovariantVector(const InputCovariantVectorType &) const override
  {
    itkExceptionMacro(<< "TransformCovariantVector(const InputCovariantVectorType &) is not implemented for "
                         "WeightedCombinationTransform");
  }


  /** This returns a sparse version of the Jacobian of the transformation.
   * In this class however, the Jacobian is not sparse.
   * However, it is a useful function, since the Jacobian is passed
   * by reference, which makes it thread-safe, unlike the normal
   * GetJacobian function. */
  void
  GetJacobian(const InputPointType & inputPoint, JacobianType & jac, NonZeroJacobianIndicesType & nzji) const override;

  /** Set the parameters. Computes the sum of weights (which is
   * the normalization term). And checks if the number of parameters
   * is correct */
  void
  SetParameters(const ParametersType & param) override;

  /** Set the fixed parameters. */
  void
  SetFixedParameters(const ParametersType &) override
  {
    // \todo: to be implemented by Stefan
  }


  /** Get the Fixed Parameters. */
  const ParametersType &
  GetFixedParameters() const override
  {
    // \todo: to be implemented by Stefan: check this:
    return this->m_FixedParameters;
  }


  /** Return the number of sub-transforms that have been set. */
  NumberOfParametersType
  GetNumberOfParameters() const override
  {
    return this->m_TransformContainer.size();
  }


  /** Set/get if the weights (parameters) should be normalized.
   * Default: false. */
  itkSetMacro(NormalizeWeights, bool);
  itkGetConstMacro(NormalizeWeights, bool);

  /** Set the vector of subtransforms. Calls a this->Modified() */
  virtual void
  SetTransformContainer(const TransformContainerType & transformContainer)
  {
    this->m_TransformContainer = transformContainer;
    this->Modified();
  }


  /** Return the vector of sub-transforms by const reference.
   * So, if you want to add a sub-transform, you should do something
   * like this:
   * TransformContainerType vec = transform->GetTransformContainer();
   * vec.push_back( newsubtransformPointer );
   * transform->SetTransformContainer( vec );
   * Although perhaps not really efficient, this makes sure that
   * this->Modified() is called when the transform container is updated.
   **/
  const TransformContainerType &
  GetTransformContainer() const
  {
    return this->m_TransformContainer;
  }


  /** Must be provided. */
  void
  GetSpatialJacobian(const InputPointType & inputPoint, SpatialJacobianType & sj) const override
  {
    itkExceptionMacro(<< "Not implemented for WeightedCombinationTransform");
  }


  void
  GetSpatialHessian(const InputPointType & inputPoint, SpatialHessianType & sh) const override
  {
    itkExceptionMacro(<< "Not implemented for WeightedCombinationTransform");
  }


  void
  GetJacobianOfSpatialJacobian(const InputPointType &          inputPoint,
                               JacobianOfSpatialJacobianType & jsj,
                               NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override
  {
    itkExceptionMacro(<< "Not implemented for WeightedCombinationTransform");
  }


  void
  GetJacobianOfSpatialJacobian(const InputPointType &          inputPoint,
                               SpatialJacobianType &           sj,
                               JacobianOfSpatialJacobianType & jsj,
                               NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override
  {
    itkExceptionMacro(<< "Not implemented for WeightedCombinationTransform");
  }


  void
  GetJacobianOfSpatialHessian(const InputPointType &         inputPoint,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override
  {
    itkExceptionMacro(<< "Not implemented for WeightedCombinationTransform");
  }


  void
  GetJacobianOfSpatialHessian(const InputPointType &         inputPoint,
                              SpatialHessianType &           sh,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override
  {
    itkExceptionMacro(<< "Not implemented for WeightedCombinationTransform");
  }


protected:
  WeightedCombinationTransform();
  ~WeightedCombinationTransform() override = default;

  TransformContainerType m_TransformContainer;
  double                 m_SumOfWeights;

  /** Precomputed nonzero Jacobian indices (simply all params) */
  NonZeroJacobianIndicesType m_NonZeroJacobianIndices;

private:
  bool m_NormalizeWeights;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkWeightedCombinationTransform.hxx"
#endif

#endif
