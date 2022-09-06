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
#ifndef itkAdvancedCombinationTransform_h
#define itkAdvancedCombinationTransform_h

#include "itkAdvancedTransform.h"
#include "itkMacro.h"

namespace itk
{

/**
 * \class AdvancedCombinationTransform
 *
 * \brief This class combines two transforms: an 'initial transform'
 * with a 'current transform'.
 *
 * The CombinationTransform class combines an initial transform \f$T_0\f$ with a
 * current transform \f$T_1\f$.
 *
 * Two methods of combining the transforms are supported:
 * \li Addition: \f$T(x) = T_0(x) + T_1(x)\f$
 * \li Composition: \f$T(x) = T_1( T_0(x) )\f$
 *
 * The TransformPoint(), the GetJacobian() and the GetInverse() methods
 * depend on this setting.
 *
 * If the transform is used in a registration framework,
 * the initial transform is assumed constant, and the current
 * transform is assumed to be the transform that is optimised.
 * So, the transform parameters of the CombinationTransform are the
 * parameters of the CurrentTransform \f$T_1\f$.
 *
 * Note: It is mandatory to set a current transform. An initial transform
 * is not mandatory.
 *
 * \ingroup Transforms
 */

template <typename TScalarType, unsigned int NDimensions = 3>
class ITK_TEMPLATE_EXPORT AdvancedCombinationTransform : public AdvancedTransform<TScalarType, NDimensions, NDimensions>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdvancedCombinationTransform);

  /** Standard itk. */
  using Self = AdvancedCombinationTransform;
  using Superclass = AdvancedTransform<TScalarType, NDimensions, NDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New method for creating an object using a factory. */
  itkNewMacro(Self);

  /** ITK Type info. */
  itkTypeMacro(AdvancedCombinationTransform, AdvancedTransform);

  /** Input and Output space dimension. */
  itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions);

  /** Typedefs inherited from Superclass.*/
  using typename Superclass::ScalarType;
  using typename Superclass::ParametersType;
  using typename Superclass::FixedParametersType;
  using typename Superclass::ParametersValueType;
  using typename Superclass::NumberOfParametersType;
  using typename Superclass::DerivativeType;
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
  using typename Superclass::InternalMatrixType;
  using InverseTransformBaseType = typename Superclass::InverseTransformBaseType;
  using typename Superclass::InverseTransformBasePointer;
  using typename Superclass::TransformCategoryEnum;
  using typename Superclass::MovingImageGradientType;
  using typename Superclass::MovingImageGradientValueType;

  /** Transform typedefs for the from Superclass. */
  using TransformType = typename Superclass::TransformType;
  using TransformTypePointer = typename TransformType::Pointer;
  using TransformTypeConstPointer = typename TransformType::ConstPointer;

  /** Typedefs for the InitialTransform. */
  using InitialTransformType = Superclass;
  using InitialTransformPointer = typename InitialTransformType::Pointer;
  using InitialTransformConstPointer = typename InitialTransformType::ConstPointer;
  using InitialTransformInverseTransformBaseType = typename InitialTransformType::InverseTransformBaseType;
  using InitialTransformInverseTransformBasePointer = typename InitialTransformType::InverseTransformBasePointer;

  /** Typedefs for the CurrentTransform. */
  using CurrentTransformType = Superclass;
  using CurrentTransformPointer = typename CurrentTransformType::Pointer;
  using CurrentTransformConstPointer = typename CurrentTransformType::ConstPointer;
  using CurrentTransformInverseTransformBaseType = typename CurrentTransformType::InverseTransformBaseType;
  using CurrentTransformInverseTransformBasePointer = typename CurrentTransformType::InverseTransformBasePointer;

  /** Set/Get a pointer to the InitialTransform. */
  void
  SetInitialTransform(InitialTransformType * _arg);

  itkGetModifiableObjectMacro(InitialTransform, InitialTransformType);

  /** Set/Get a pointer to the CurrentTransform.
   * Make sure to set the CurrentTransform before calling functions like
   * TransformPoint(), GetJacobian(), SetParameters() etc.
   */
  void
  SetCurrentTransform(CurrentTransformType * _arg);

  itkGetModifiableObjectMacro(CurrentTransform, CurrentTransformType);

  /** Return the number of sub-transforms. */
  SizeValueType
  GetNumberOfTransforms() const;

  /** Get the Nth current transform.
   * Exact interface to the ITK4 MultiTransform::GetNthTransform( SizeValueType n )
   * \warning The bounds checking is performed.
   */
  const TransformTypePointer
  GetNthTransform(SizeValueType n) const;

  /** Control the way transforms are combined. */
  void
  SetUseComposition(bool _arg);

  itkGetConstMacro(UseComposition, bool);

  /** Control the way transforms are combined. */
  void
  SetUseAddition(bool _arg);

  itkGetConstMacro(UseAddition, bool);

  /**  Method to transform a point. */
  OutputPointType
  TransformPoint(const InputPointType & point) const override;

  /** ITK4 change:
   * The following pure virtual functions must be overloaded.
   * For now just throw an exception, since these are not used in elastix.
   */
  OutputVectorType
  TransformVector(const InputVectorType &) const override
  {
    itkExceptionMacro(
      << "TransformVector(const InputVectorType &) is not implemented for AdvancedCombinationTransform");
  }


  OutputVnlVectorType
  TransformVector(const InputVnlVectorType &) const override
  {
    itkExceptionMacro(
      << "TransformVector(const InputVnlVectorType &) is not implemented for AdvancedCombinationTransform");
  }


  OutputCovariantVectorType
  TransformCovariantVector(const InputCovariantVectorType &) const override
  {
    itkExceptionMacro(<< "TransformCovariantVector(const InputCovariantVectorType &) is not implemented for "
                         "AdvancedCombinationTransform");
  }


  /** Return the number of parameters that completely define the CurrentTransform. */
  NumberOfParametersType
  GetNumberOfParameters() const override;

  /** Get the number of nonzero Jacobian indices. By default all. */
  NumberOfParametersType
  GetNumberOfNonZeroJacobianIndices() const override;

  /** Get the transformation parameters from the CurrentTransform. */
  const ParametersType &
  GetParameters() const override;

  /** Get the fixed parameters from the CurrentTransform. */
  const FixedParametersType &
  GetFixedParameters() const override;

  /** Set the transformation parameters in the CurrentTransform. */
  void
  SetParameters(const ParametersType & param) override;

  /** Set the transformation parameters in the CurrentTransform.
   * This method forces the transform to copy the parameters.
   */
  void
  SetParametersByValue(const ParametersType & param) override;

  /** Set the fixed parameters in the CurrentTransform. */
  void
  SetFixedParameters(const FixedParametersType & fixedParam) override;

  /** Return the inverse \f$T^{-1}\f$ of the transform.
   *  This is only possible when:
   * - both the inverses of the initial and the current transform
   *   are defined, and Composition is used:
   *   \f$T^{-1}(y) = T_0^{-1} ( T_1^{-1}(y) )\f$
   * - No initial transform is used and the current transform is defined.
   * In all other cases this function returns false and does not provide
   * an inverse transform. An exception is thrown when no CurrentTransform
   * is set.
   */
  bool
  GetInverse(Self * inverse) const;

  /** Return whether the transform is linear (or actually: affine)
   * Returns true when both initial and current transform are linear */
  bool
  IsLinear() const override;

  /** Special handling for combination transform. If all transforms
   * are linear, then return category Linear. Otherwise if all
   * transforms set to optimize are DisplacementFields, then
   * return DisplacementField category. */
  TransformCategoryEnum
  GetTransformCategory() const override;

  /** Whether the advanced transform has nonzero matrices. */
  bool
  GetHasNonZeroSpatialHessian() const override;

  bool
  HasNonZeroJacobianOfSpatialHessian() const;

  /** Compute the (sparse) Jacobian of the transformation. */
  void
  GetJacobian(const InputPointType &       inputPoint,
              JacobianType &               j,
              NonZeroJacobianIndicesType & nonZeroJacobianIndices) const override;

  /** Compute the inner product of the Jacobian with the moving image gradient. */
  void
  EvaluateJacobianWithImageGradientProduct(const InputPointType &          inputPoint,
                                           const MovingImageGradientType & movingImageGradient,
                                           DerivativeType &                imageJacobian,
                                           NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override;

  /** Compute the spatial Jacobian of the transformation. */
  void
  GetSpatialJacobian(const InputPointType & inputPoint, SpatialJacobianType & sj) const override;

  /** Compute the spatial Hessian of the transformation. */
  void
  GetSpatialHessian(const InputPointType & inputPoint, SpatialHessianType & sh) const override;

  /** Compute the Jacobian of the spatial Jacobian of the transformation. */
  void
  GetJacobianOfSpatialJacobian(const InputPointType &          inputPoint,
                               JacobianOfSpatialJacobianType & jsj,
                               NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override;

  /** Compute both the spatial Jacobian and the Jacobian of the
   * spatial Jacobian of the transformation.
   */
  void
  GetJacobianOfSpatialJacobian(const InputPointType &          inputPoint,
                               SpatialJacobianType &           sj,
                               JacobianOfSpatialJacobianType & jsj,
                               NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override;

  /** Compute the Jacobian of the spatial Hessian of the transformation. */
  void
  GetJacobianOfSpatialHessian(const InputPointType &         inputPoint,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override;

  /** Compute both the spatial Hessian and the Jacobian of the
   * spatial Hessian of the transformation.
   */
  void
  GetJacobianOfSpatialHessian(const InputPointType &         inputPoint,
                              SpatialHessianType &           sh,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override;

protected:
  /** Constructor. */
  AdvancedCombinationTransform();

  /** Destructor. */
  ~AdvancedCombinationTransform() override = default;

  /** Set the SelectedTransformPointFunction and the
   * SelectedGetJacobianFunction.
   */
  void
  UpdateCombinationMethod();

  /** ************************************************
   * Methods to transform a point.
   */

  /** ADDITION: \f$T(x) = T_0(x) + T_1(x) - x\f$ */
  inline OutputPointType
  TransformPointUseAddition(const InputPointType & point) const;

  /** COMPOSITION: \f$T(x) = T_1( T_0(x) )\f$
   * \warning: assumes that input and output point type are the same.
   */
  inline OutputPointType
  TransformPointUseComposition(const InputPointType & point) const;

  /** CURRENT ONLY: \f$T(x) = T_1(x)\f$ */
  inline OutputPointType
  TransformPointNoInitialTransform(const InputPointType & point) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline OutputPointType
  TransformPointNoCurrentTransform(const InputPointType & point) const;

  /** ************************************************
   * Methods to compute the sparse Jacobian.
   */

  /** ADDITION: \f$J(x) = J_1(x)\f$ */
  inline void
  GetJacobianUseAddition(const InputPointType &, JacobianType &, NonZeroJacobianIndicesType &) const;

  /** COMPOSITION: \f$J(x) = J_1( T_0(x) )\f$
   * \warning: assumes that input and output point type are the same.
   */
  inline void
  GetJacobianUseComposition(const InputPointType &, JacobianType &, NonZeroJacobianIndicesType &) const;

  /** CURRENT ONLY: \f$J(x) = J_1(x)\f$ */
  inline void
  GetJacobianNoInitialTransform(const InputPointType &, JacobianType &, NonZeroJacobianIndicesType &) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline void
  GetJacobianNoCurrentTransform(const InputPointType &, JacobianType &, NonZeroJacobianIndicesType &) const;

  /** ************************************************
   * Methods to compute the inner product of the Jacobian with the moving image gradient.
   */

  /** ADDITION: \f$J(x) = J_1(x)\f$ */
  inline void
  EvaluateJacobianWithImageGradientProductUseAddition(const InputPointType &,
                                                      const MovingImageGradientType &,
                                                      DerivativeType &,
                                                      NonZeroJacobianIndicesType &) const;

  /** COMPOSITION: \f$J(x) = J_1( T_0(x) )\f$
   * \warning: assumes that input and output point type are the same.
   */
  inline void
  EvaluateJacobianWithImageGradientProductUseComposition(const InputPointType &,
                                                         const MovingImageGradientType &,
                                                         DerivativeType &,
                                                         NonZeroJacobianIndicesType &) const;

  /** CURRENT ONLY: \f$J(x) = J_1(x)\f$ */
  inline void
  EvaluateJacobianWithImageGradientProductNoInitialTransform(const InputPointType &,
                                                             const MovingImageGradientType &,
                                                             DerivativeType &,
                                                             NonZeroJacobianIndicesType &) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline void
  EvaluateJacobianWithImageGradientProductNoCurrentTransform(const InputPointType &,
                                                             const MovingImageGradientType &,
                                                             DerivativeType &,
                                                             NonZeroJacobianIndicesType &) const;

  /** ************************************************
   * Methods to compute the spatial Jacobian.
   */

  /** ADDITION: \f$J(x) = J_1(x)\f$ */
  inline void
  GetSpatialJacobianUseAddition(const InputPointType & inputPoint, SpatialJacobianType & sj) const;

  /** COMPOSITION: \f$J(x) = J_1( T_0(x) )\f$
   * \warning: assumes that input and output point type are the same.
   */
  inline void
  GetSpatialJacobianUseComposition(const InputPointType & inputPoint, SpatialJacobianType & sj) const;

  /** CURRENT ONLY: \f$J(x) = J_1(x)\f$ */
  inline void
  GetSpatialJacobianNoInitialTransform(const InputPointType & inputPoint, SpatialJacobianType & sj) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline void
  GetSpatialJacobianNoCurrentTransform(const InputPointType & inputPoint, SpatialJacobianType & sj) const;

  /** ************************************************
   * Methods to compute the spatial Hessian.
   */

  /** ADDITION: \f$J(x) = J_1(x)\f$ */
  inline void
  GetSpatialHessianUseAddition(const InputPointType & inputPoint, SpatialHessianType & sh) const;

  /** COMPOSITION: \f$J(x) = J_1( T_0(x) )\f$
   * \warning: assumes that input and output point type are the same.
   */
  inline void
  GetSpatialHessianUseComposition(const InputPointType & inputPoint, SpatialHessianType & sh) const;

  /** CURRENT ONLY: \f$J(x) = J_1(x)\f$ */
  inline void
  GetSpatialHessianNoInitialTransform(const InputPointType & inputPoint, SpatialHessianType & sh) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline void
  GetSpatialHessianNoCurrentTransform(const InputPointType & inputPoint, SpatialHessianType & sh) const;

  /** ************************************************
   * Methods to compute the Jacobian of the spatial Jacobian.
   */

  /** ADDITION: \f$J(x) = J_1(x)\f$ */
  inline void
  GetJacobianOfSpatialJacobianUseAddition(const InputPointType &          inputPoint,
                                          JacobianOfSpatialJacobianType & jsj,
                                          NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const;

  inline void
  GetJacobianOfSpatialJacobianUseAddition(const InputPointType &          inputPoint,
                                          SpatialJacobianType &           sj,
                                          JacobianOfSpatialJacobianType & jsj,
                                          NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const;

  /** COMPOSITION: \f$J(x) = J_1( T_0(x) )\f$
   * \warning: assumes that input and output point type are the same.
   */
  inline void
  GetJacobianOfSpatialJacobianUseComposition(const InputPointType &          inputPoint,
                                             JacobianOfSpatialJacobianType & jsj,
                                             NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const;

  inline void
  GetJacobianOfSpatialJacobianUseComposition(const InputPointType &          inputPoint,
                                             SpatialJacobianType &           sj,
                                             JacobianOfSpatialJacobianType & jsj,
                                             NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const;

  /** CURRENT ONLY: \f$J(x) = J_1(x)\f$ */
  inline void
  GetJacobianOfSpatialJacobianNoInitialTransform(const InputPointType &          inputPoint,
                                                 JacobianOfSpatialJacobianType & jsj,
                                                 NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const;

  inline void
  GetJacobianOfSpatialJacobianNoInitialTransform(const InputPointType &          inputPoint,
                                                 SpatialJacobianType &           sj,
                                                 JacobianOfSpatialJacobianType & jsj,
                                                 NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline void
  GetJacobianOfSpatialJacobianNoCurrentTransform(const InputPointType &          inputPoint,
                                                 JacobianOfSpatialJacobianType & jsj,
                                                 NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const;

  inline void
  GetJacobianOfSpatialJacobianNoCurrentTransform(const InputPointType &          inputPoint,
                                                 SpatialJacobianType &           sj,
                                                 JacobianOfSpatialJacobianType & jsj,
                                                 NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const;

  /** ************************************************
   * Methods to compute the Jacobian of the spatial Hessian.
   */

  /** ADDITION: \f$J(x) = J_1(x)\f$ */
  inline void
  GetJacobianOfSpatialHessianUseAddition(const InputPointType &         inputPoint,
                                         JacobianOfSpatialHessianType & jsh,
                                         NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const;

  inline void
  GetJacobianOfSpatialHessianUseAddition(const InputPointType &         inputPoint,
                                         SpatialHessianType &           sh,
                                         JacobianOfSpatialHessianType & jsh,
                                         NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const;

  /** COMPOSITION: \f$J(x) = J_1( T_0(x) )\f$
   * \warning: assumes that input and output point type are the same.
   */
  inline void
  GetJacobianOfSpatialHessianUseComposition(const InputPointType &         inputPoint,
                                            JacobianOfSpatialHessianType & jsh,
                                            NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const;

  inline void
  GetJacobianOfSpatialHessianUseComposition(const InputPointType &         inputPoint,
                                            SpatialHessianType &           sh,
                                            JacobianOfSpatialHessianType & jsh,
                                            NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const;

  /** CURRENT ONLY: \f$J(x) = J_1(x)\f$ */
  inline void
  GetJacobianOfSpatialHessianNoInitialTransform(const InputPointType &         inputPoint,
                                                JacobianOfSpatialHessianType & jsh,
                                                NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const;

  inline void
  GetJacobianOfSpatialHessianNoInitialTransform(const InputPointType &         inputPoint,
                                                SpatialHessianType &           sh,
                                                JacobianOfSpatialHessianType & jsh,
                                                NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline void
  GetJacobianOfSpatialHessianNoCurrentTransform(const InputPointType &         inputPoint,
                                                JacobianOfSpatialHessianType & jsh,
                                                NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const;

  inline void
  GetJacobianOfSpatialHessianNoCurrentTransform(const InputPointType &         inputPoint,
                                                SpatialHessianType &           sh,
                                                JacobianOfSpatialHessianType & jsh,
                                                NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const;

private:
  /** Exception text. */
  constexpr static const char * NoCurrentTransformSet = "No current transform set in the AdvancedCombinationTransform";

  /** Declaration of members. */
  InitialTransformPointer m_InitialTransform{ nullptr };
  CurrentTransformPointer m_CurrentTransform{ nullptr };

  /** Typedefs for function pointers. */
  using TransformPointFunctionPointer = OutputPointType (Self::*)(const InputPointType &) const;
  using GetSparseJacobianFunctionPointer = void (Self::*)(const InputPointType &,
                                                          JacobianType &,
                                                          NonZeroJacobianIndicesType &) const;
  using EvaluateJacobianWithImageGradientProductFunctionPointer = void (Self::*)(const InputPointType &,
                                                                                 const MovingImageGradientType &,
                                                                                 DerivativeType &,
                                                                                 NonZeroJacobianIndicesType &) const;
  using GetSpatialJacobianFunctionPointer = void (Self::*)(const InputPointType &, SpatialJacobianType &) const;
  using GetSpatialHessianFunctionPointer = void (Self::*)(const InputPointType &, SpatialHessianType &) const;
  using GetJacobianOfSpatialJacobianFunctionPointer = void (Self::*)(const InputPointType &,
                                                                     JacobianOfSpatialJacobianType &,
                                                                     NonZeroJacobianIndicesType &) const;
  using GetJacobianOfSpatialJacobianFunctionPointer2 = void (Self::*)(const InputPointType &,
                                                                      SpatialJacobianType &,
                                                                      JacobianOfSpatialJacobianType &,
                                                                      NonZeroJacobianIndicesType &) const;
  using GetJacobianOfSpatialHessianFunctionPointer = void (Self::*)(const InputPointType &,
                                                                    JacobianOfSpatialHessianType &,
                                                                    NonZeroJacobianIndicesType &) const;
  using GetJacobianOfSpatialHessianFunctionPointer2 = void (Self::*)(const InputPointType &,
                                                                     SpatialHessianType &,
                                                                     JacobianOfSpatialHessianType &,
                                                                     NonZeroJacobianIndicesType &) const;

  /**  A pointer to one of the following functions:
   * - TransformPointUseAddition,
   * - TransformPointUseComposition,
   * - TransformPointNoCurrentTransform
   * - TransformPointNoInitialTransform.
   */
  TransformPointFunctionPointer m_SelectedTransformPointFunction{ &Self::TransformPointNoCurrentTransform };

  /**  A pointer to one of the following functions:
   * - GetJacobianUseAddition,
   * - GetJacobianUseComposition,
   * - GetJacobianNoCurrentTransform
   * - GetJacobianNoInitialTransform.
   */
  // GetJacobianFunctionPointer m_SelectedGetJacobianFunction;

  /** More of these. Set everything to have no current transform. */
  GetSparseJacobianFunctionPointer m_SelectedGetSparseJacobianFunction{ &Self::GetJacobianNoCurrentTransform };
  EvaluateJacobianWithImageGradientProductFunctionPointer m_SelectedEvaluateJacobianWithImageGradientProductFunction{
    &Self::EvaluateJacobianWithImageGradientProductNoInitialTransform
  };
  GetSpatialJacobianFunctionPointer m_SelectedGetSpatialJacobianFunction{ &Self::GetSpatialJacobianNoCurrentTransform };
  GetSpatialHessianFunctionPointer  m_SelectedGetSpatialHessianFunction{ &Self::GetSpatialHessianNoCurrentTransform };
  GetJacobianOfSpatialJacobianFunctionPointer m_SelectedGetJacobianOfSpatialJacobianFunction{
    &Self::GetJacobianOfSpatialJacobianNoCurrentTransform
  };
  GetJacobianOfSpatialJacobianFunctionPointer2 m_SelectedGetJacobianOfSpatialJacobianFunction2{
    &Self::GetJacobianOfSpatialJacobianNoCurrentTransform
  };
  GetJacobianOfSpatialHessianFunctionPointer m_SelectedGetJacobianOfSpatialHessianFunction{
    &Self::GetJacobianOfSpatialHessianNoCurrentTransform
  };
  GetJacobianOfSpatialHessianFunctionPointer2 m_SelectedGetJacobianOfSpatialHessianFunction2{
    &Self::GetJacobianOfSpatialHessianNoCurrentTransform
  };

  /** How to combine the transformations. Composition by default. */
  bool m_UseAddition{ false };
  bool m_UseComposition{ true };
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAdvancedCombinationTransform.hxx"
#endif

#endif // end #ifndef itkAdvancedCombinationTransform_h
