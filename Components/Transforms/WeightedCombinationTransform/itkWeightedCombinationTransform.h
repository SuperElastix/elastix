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
class WeightedCombinationTransform : public AdvancedTransform<TScalarType, NInputDimensions, NOutputDimensions>
{
public:
  /** Standard class typedefs. */
  typedef WeightedCombinationTransform                                        Self;
  typedef AdvancedTransform<TScalarType, NInputDimensions, NOutputDimensions> Superclass;
  typedef SmartPointer<Self>                                                  Pointer;
  typedef SmartPointer<const Self>                                            ConstPointer;

  /** New method for creating an object using a factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(WeightedCombinationTransform, AdvancedTransform);

  /** Dimension of the domain space. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, NInputDimensions);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, NOutputDimensions);

  /** Typedefs from the Superclass. */
  typedef typename Superclass::ScalarType                    ScalarType;
  typedef typename Superclass::ParametersType                ParametersType;
  typedef typename Superclass::NumberOfParametersType        NumberOfParametersType;
  typedef typename Superclass::JacobianType                  JacobianType;
  typedef typename Superclass::InputVectorType               InputVectorType;
  typedef typename Superclass::OutputVectorType              OutputVectorType;
  typedef typename Superclass ::InputCovariantVectorType     InputCovariantVectorType;
  typedef typename Superclass ::OutputCovariantVectorType    OutputCovariantVectorType;
  typedef typename Superclass::InputVnlVectorType            InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType           OutputVnlVectorType;
  typedef typename Superclass::InputPointType                InputPointType;
  typedef typename Superclass::OutputPointType               OutputPointType;
  typedef typename Superclass::NonZeroJacobianIndicesType    NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType           SpatialJacobianType;
  typedef typename Superclass::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType            SpatialHessianType;
  typedef typename Superclass ::JacobianOfSpatialHessianType JacobianOfSpatialHessianType;

  /** New typedefs in this class: */
  typedef Transform<TScalarType, NInputDimensions, NOutputDimensions> TransformType;
  /** \todo: shouldn't these be ConstPointers? */
  typedef typename TransformType::Pointer TransformPointer;
  typedef std::vector<TransformPointer>   TransformContainerType;

  /**  Method to transform a point. */
  OutputPointType
  TransformPoint(const InputPointType & ipp) const override;

  /** These vector transforms are not implemented for this transform. */
  OutputVectorType
  TransformVector(const InputVectorType &) const override
  {
    itkExceptionMacro(<< "TransformVector(const InputVectorType &) is not implemented "
                      << "for WeightedCombinationTransform");
  }


  OutputVnlVectorType
  TransformVector(const InputVnlVectorType &) const override
  {
    itkExceptionMacro(<< "TransformVector(const InputVnlVectorType &) is not implemented "
                      << "for WeightedCombinationTransform");
  }


  OutputCovariantVectorType
  TransformCovariantVector(const InputCovariantVectorType &) const override
  {
    itkExceptionMacro(<< "TransformCovariantVector(const InputCovariantVectorType &) is not implemented "
                      << "for WeightedCombinationTransform");
  }


  /** This returns a sparse version of the Jacobian of the transformation.
   * In this class however, the Jacobian is not sparse.
   * However, it is a useful function, since the Jacobian is passed
   * by reference, which makes it thread-safe, unlike the normal
   * GetJacobian function. */
  void
  GetJacobian(const InputPointType & ipp, JacobianType & jac, NonZeroJacobianIndicesType & nzji) const override;

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
  GetFixedParameters(void) const override
  {
    // \todo: to be implemented by Stefan: check this:
    return this->m_FixedParameters;
  }


  /** Return the number of sub-transforms that have been set. */
  NumberOfParametersType
  GetNumberOfParameters(void) const override
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
  GetTransformContainer(void) const
  {
    return this->m_TransformContainer;
  }


  /** Must be provided. */
  void
  GetSpatialJacobian(const InputPointType & ipp, SpatialJacobianType & sj) const override
  {
    itkExceptionMacro(<< "Not implemented for WeightedCombinationTransform");
  }


  void
  GetSpatialHessian(const InputPointType & ipp, SpatialHessianType & sh) const override
  {
    itkExceptionMacro(<< "Not implemented for WeightedCombinationTransform");
  }


  void
  GetJacobianOfSpatialJacobian(const InputPointType &          ipp,
                               JacobianOfSpatialJacobianType & jsj,
                               NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override
  {
    itkExceptionMacro(<< "Not implemented for WeightedCombinationTransform");
  }


  void
  GetJacobianOfSpatialJacobian(const InputPointType &          ipp,
                               SpatialJacobianType &           sj,
                               JacobianOfSpatialJacobianType & jsj,
                               NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override
  {
    itkExceptionMacro(<< "Not implemented for WeightedCombinationTransform");
  }


  void
  GetJacobianOfSpatialHessian(const InputPointType &         ipp,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override
  {
    itkExceptionMacro(<< "Not implemented for WeightedCombinationTransform");
  }


  void
  GetJacobianOfSpatialHessian(const InputPointType &         ipp,
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
  WeightedCombinationTransform(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  bool m_NormalizeWeights;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkWeightedCombinationTransform.hxx"
#endif

#endif
