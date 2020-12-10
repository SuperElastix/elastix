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
class AdvancedCombinationTransform : public AdvancedTransform<TScalarType, NDimensions, NDimensions>
{
public:
  /** Standard itk. */
  typedef AdvancedCombinationTransform                             Self;
  typedef AdvancedTransform<TScalarType, NDimensions, NDimensions> Superclass;
  typedef SmartPointer<Self>                                       Pointer;
  typedef SmartPointer<const Self>                                 ConstPointer;

  /** New method for creating an object using a factory. */
  itkNewMacro(Self);

  /** ITK Type info. */
  itkTypeMacro(AdvancedCombinationTransform, AdvancedTransform);

  /** Input and Output space dimension. */
  itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions);

  /** Typedefs inherited from Superclass.*/
  typedef typename Superclass::ScalarType                    ScalarType;
  typedef typename Superclass::ParametersType                ParametersType;
  typedef typename Superclass::FixedParametersType           FixedParametersType;
  typedef typename Superclass::ParametersValueType           ParametersValueType;
  typedef typename Superclass::NumberOfParametersType        NumberOfParametersType;
  typedef typename Superclass::DerivativeType                DerivativeType;
  typedef typename Superclass::JacobianType                  JacobianType;
  typedef typename Superclass::InputVectorType               InputVectorType;
  typedef typename Superclass::OutputVectorType              OutputVectorType;
  typedef typename Superclass::InputCovariantVectorType      InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType     OutputCovariantVectorType;
  typedef typename Superclass::InputVnlVectorType            InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType           OutputVnlVectorType;
  typedef typename Superclass::InputPointType                InputPointType;
  typedef typename Superclass::OutputPointType               OutputPointType;
  typedef typename Superclass::NonZeroJacobianIndicesType    NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType           SpatialJacobianType;
  typedef typename Superclass::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType            SpatialHessianType;
  typedef typename Superclass::JacobianOfSpatialHessianType  JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType            InternalMatrixType;
  typedef typename Superclass::InverseTransformBaseType      InverseTransformBaseType;
  typedef typename Superclass::InverseTransformBasePointer   InverseTransformBasePointer;
  typedef typename Superclass::TransformCategoryEnum         TransformCategoryEnum;
  typedef typename Superclass::MovingImageGradientType       MovingImageGradientType;
  typedef typename Superclass::MovingImageGradientValueType  MovingImageGradientValueType;

  /** Transform typedefs for the from Superclass. */
  typedef typename Superclass::TransformType   TransformType;
  typedef typename TransformType::Pointer      TransformTypePointer;
  typedef typename TransformType::ConstPointer TransformTypeConstPointer;

  /** Typedefs for the InitialTransform. */
  typedef Superclass                                                 InitialTransformType;
  typedef typename InitialTransformType::Pointer                     InitialTransformPointer;
  typedef typename InitialTransformType::ConstPointer                InitialTransformConstPointer;
  typedef typename InitialTransformType::InverseTransformBaseType    InitialTransformInverseTransformBaseType;
  typedef typename InitialTransformType::InverseTransformBasePointer InitialTransformInverseTransformBasePointer;

  /** Typedefs for the CurrentTransform. */
  typedef Superclass                                                 CurrentTransformType;
  typedef typename CurrentTransformType::Pointer                     CurrentTransformPointer;
  typedef typename CurrentTransformType::ConstPointer                CurrentTransformConstPointer;
  typedef typename CurrentTransformType::InverseTransformBaseType    CurrentTransformInverseTransformBaseType;
  typedef typename CurrentTransformType::InverseTransformBasePointer CurrentTransformInverseTransformBasePointer;

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
  GetNumberOfTransforms(void) const;

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
    itkExceptionMacro(<< "TransformVector(const InputVectorType &) is not implemented "
                      << "for AdvancedCombinationTransform");
  }


  OutputVnlVectorType
  TransformVector(const InputVnlVectorType &) const override
  {
    itkExceptionMacro(<< "TransformVector(const InputVnlVectorType &) is not implemented "
                      << "for AdvancedCombinationTransform");
  }


  OutputCovariantVectorType
  TransformCovariantVector(const InputCovariantVectorType &) const override
  {
    itkExceptionMacro(<< "TransformCovariantVector(const InputCovariantVectorType &) is not implemented "
                      << "for AdvancedCombinationTransform");
  }


  /** Return the number of parameters that completely define the CurrentTransform. */
  NumberOfParametersType
  GetNumberOfParameters(void) const override;

  /** Get the number of nonzero Jacobian indices. By default all. */
  NumberOfParametersType
  GetNumberOfNonZeroJacobianIndices(void) const override;

  /** Get the transformation parameters from the CurrentTransform. */
  const ParametersType &
  GetParameters(void) const override;

  /** Get the fixed parameters from the CurrentTransform. */
  const FixedParametersType &
  GetFixedParameters(void) const override;

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
  IsLinear(void) const override;

  /** Special handling for combination transform. If all transforms
   * are linear, then return category Linear. Otherwise if all
   * transforms set to optimize are DisplacementFields, then
   * return DisplacementField category. */
  TransformCategoryEnum
  GetTransformCategory() const override;

  /** Whether the advanced transform has nonzero matrices. */
  bool
  GetHasNonZeroSpatialHessian(void) const override;

  bool
  HasNonZeroJacobianOfSpatialHessian(void) const;

  /** Compute the (sparse) Jacobian of the transformation. */
  void
  GetJacobian(const InputPointType &       ipp,
              JacobianType &               j,
              NonZeroJacobianIndicesType & nonZeroJacobianIndices) const override;

  /** Compute the inner product of the Jacobian with the moving image gradient. */
  void
  EvaluateJacobianWithImageGradientProduct(const InputPointType &          ipp,
                                           const MovingImageGradientType & movingImageGradient,
                                           DerivativeType &                imageJacobian,
                                           NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override;

  /** Compute the spatial Jacobian of the transformation. */
  void
  GetSpatialJacobian(const InputPointType & ipp, SpatialJacobianType & sj) const override;

  /** Compute the spatial Hessian of the transformation. */
  void
  GetSpatialHessian(const InputPointType & ipp, SpatialHessianType & sh) const override;

  /** Compute the Jacobian of the spatial Jacobian of the transformation. */
  void
  GetJacobianOfSpatialJacobian(const InputPointType &          ipp,
                               JacobianOfSpatialJacobianType & jsj,
                               NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override;

  /** Compute both the spatial Jacobian and the Jacobian of the
   * spatial Jacobian of the transformation.
   */
  void
  GetJacobianOfSpatialJacobian(const InputPointType &          ipp,
                               SpatialJacobianType &           sj,
                               JacobianOfSpatialJacobianType & jsj,
                               NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override;

  /** Compute the Jacobian of the spatial Hessian of the transformation. */
  void
  GetJacobianOfSpatialHessian(const InputPointType &         ipp,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override;

  /** Compute both the spatial Hessian and the Jacobian of the
   * spatial Hessian of the transformation.
   */
  void
  GetJacobianOfSpatialHessian(const InputPointType &         ipp,
                              SpatialHessianType &           sh,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override;

  /** Typedefs for function pointers. */
  typedef OutputPointType (Self::*TransformPointFunctionPointer)(const InputPointType &) const;
  typedef void (Self::*GetSparseJacobianFunctionPointer)(const InputPointType &,
                                                         JacobianType &,
                                                         NonZeroJacobianIndicesType &) const;
  typedef void (Self::*EvaluateJacobianWithImageGradientProductFunctionPointer)(const InputPointType &,
                                                                                const MovingImageGradientType &,
                                                                                DerivativeType &,
                                                                                NonZeroJacobianIndicesType &) const;
  typedef void (Self::*GetSpatialJacobianFunctionPointer)(const InputPointType &, SpatialJacobianType &) const;
  typedef void (Self::*GetSpatialHessianFunctionPointer)(const InputPointType &, SpatialHessianType &) const;
  typedef void (Self::*GetJacobianOfSpatialJacobianFunctionPointer)(const InputPointType &,
                                                                    JacobianOfSpatialJacobianType &,
                                                                    NonZeroJacobianIndicesType &) const;
  typedef void (Self::*GetJacobianOfSpatialJacobianFunctionPointer2)(const InputPointType &,
                                                                     SpatialJacobianType &,
                                                                     JacobianOfSpatialJacobianType &,
                                                                     NonZeroJacobianIndicesType &) const;
  typedef void (Self::*GetJacobianOfSpatialHessianFunctionPointer)(const InputPointType &,
                                                                   JacobianOfSpatialHessianType &,
                                                                   NonZeroJacobianIndicesType &) const;
  typedef void (Self::*GetJacobianOfSpatialHessianFunctionPointer2)(const InputPointType &,
                                                                    SpatialHessianType &,
                                                                    JacobianOfSpatialHessianType &,
                                                                    NonZeroJacobianIndicesType &) const;

protected:
  /** Constructor. */
  AdvancedCombinationTransform();

  /** Destructor. */
  ~AdvancedCombinationTransform() override = default;

  /** Set the SelectedTransformPointFunction and the
   * SelectedGetJacobianFunction.
   */
  void
  UpdateCombinationMethod(void);

  /** Throw an exception. */
  void
  NoCurrentTransformSet(void) const;

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
  GetSpatialJacobianUseAddition(const InputPointType & ipp, SpatialJacobianType & sj) const;

  /** COMPOSITION: \f$J(x) = J_1( T_0(x) )\f$
   * \warning: assumes that input and output point type are the same.
   */
  inline void
  GetSpatialJacobianUseComposition(const InputPointType & ipp, SpatialJacobianType & sj) const;

  /** CURRENT ONLY: \f$J(x) = J_1(x)\f$ */
  inline void
  GetSpatialJacobianNoInitialTransform(const InputPointType & ipp, SpatialJacobianType & sj) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline void
  GetSpatialJacobianNoCurrentTransform(const InputPointType & ipp, SpatialJacobianType & sj) const;

  /** ************************************************
   * Methods to compute the spatial Hessian.
   */

  /** ADDITION: \f$J(x) = J_1(x)\f$ */
  inline void
  GetSpatialHessianUseAddition(const InputPointType & ipp, SpatialHessianType & sh) const;

  /** COMPOSITION: \f$J(x) = J_1( T_0(x) )\f$
   * \warning: assumes that input and output point type are the same.
   */
  inline void
  GetSpatialHessianUseComposition(const InputPointType & ipp, SpatialHessianType & sh) const;

  /** CURRENT ONLY: \f$J(x) = J_1(x)\f$ */
  inline void
  GetSpatialHessianNoInitialTransform(const InputPointType & ipp, SpatialHessianType & sh) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline void
  GetSpatialHessianNoCurrentTransform(const InputPointType & ipp, SpatialHessianType & sh) const;

  /** ************************************************
   * Methods to compute the Jacobian of the spatial Jacobian.
   */

  /** ADDITION: \f$J(x) = J_1(x)\f$ */
  inline void
  GetJacobianOfSpatialJacobianUseAddition(const InputPointType &          ipp,
                                          JacobianOfSpatialJacobianType & jsj,
                                          NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const;

  inline void
  GetJacobianOfSpatialJacobianUseAddition(const InputPointType &          ipp,
                                          SpatialJacobianType &           sj,
                                          JacobianOfSpatialJacobianType & jsj,
                                          NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const;

  /** COMPOSITION: \f$J(x) = J_1( T_0(x) )\f$
   * \warning: assumes that input and output point type are the same.
   */
  inline void
  GetJacobianOfSpatialJacobianUseComposition(const InputPointType &          ipp,
                                             JacobianOfSpatialJacobianType & jsj,
                                             NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const;

  inline void
  GetJacobianOfSpatialJacobianUseComposition(const InputPointType &          ipp,
                                             SpatialJacobianType &           sj,
                                             JacobianOfSpatialJacobianType & jsj,
                                             NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const;

  /** CURRENT ONLY: \f$J(x) = J_1(x)\f$ */
  inline void
  GetJacobianOfSpatialJacobianNoInitialTransform(const InputPointType &          ipp,
                                                 JacobianOfSpatialJacobianType & jsj,
                                                 NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const;

  inline void
  GetJacobianOfSpatialJacobianNoInitialTransform(const InputPointType &          ipp,
                                                 SpatialJacobianType &           sj,
                                                 JacobianOfSpatialJacobianType & jsj,
                                                 NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline void
  GetJacobianOfSpatialJacobianNoCurrentTransform(const InputPointType &          ipp,
                                                 JacobianOfSpatialJacobianType & jsj,
                                                 NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const;

  inline void
  GetJacobianOfSpatialJacobianNoCurrentTransform(const InputPointType &          ipp,
                                                 SpatialJacobianType &           sj,
                                                 JacobianOfSpatialJacobianType & jsj,
                                                 NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const;

  /** ************************************************
   * Methods to compute the Jacobian of the spatial Hessian.
   */

  /** ADDITION: \f$J(x) = J_1(x)\f$ */
  inline void
  GetJacobianOfSpatialHessianUseAddition(const InputPointType &         ipp,
                                         JacobianOfSpatialHessianType & jsh,
                                         NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const;

  inline void
  GetJacobianOfSpatialHessianUseAddition(const InputPointType &         ipp,
                                         SpatialHessianType &           sh,
                                         JacobianOfSpatialHessianType & jsh,
                                         NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const;

  /** COMPOSITION: \f$J(x) = J_1( T_0(x) )\f$
   * \warning: assumes that input and output point type are the same.
   */
  inline void
  GetJacobianOfSpatialHessianUseComposition(const InputPointType &         ipp,
                                            JacobianOfSpatialHessianType & jsh,
                                            NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const;

  inline void
  GetJacobianOfSpatialHessianUseComposition(const InputPointType &         ipp,
                                            SpatialHessianType &           sh,
                                            JacobianOfSpatialHessianType & jsh,
                                            NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const;

  /** CURRENT ONLY: \f$J(x) = J_1(x)\f$ */
  inline void
  GetJacobianOfSpatialHessianNoInitialTransform(const InputPointType &         ipp,
                                                JacobianOfSpatialHessianType & jsh,
                                                NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const;

  inline void
  GetJacobianOfSpatialHessianNoInitialTransform(const InputPointType &         ipp,
                                                SpatialHessianType &           sh,
                                                JacobianOfSpatialHessianType & jsh,
                                                NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline void
  GetJacobianOfSpatialHessianNoCurrentTransform(const InputPointType &         ipp,
                                                JacobianOfSpatialHessianType & jsh,
                                                NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const;

  inline void
  GetJacobianOfSpatialHessianNoCurrentTransform(const InputPointType &         ipp,
                                                SpatialHessianType &           sh,
                                                JacobianOfSpatialHessianType & jsh,
                                                NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const;

private:
  /** Declaration of members. */
  InitialTransformPointer m_InitialTransform;
  CurrentTransformPointer m_CurrentTransform;

  /**  A pointer to one of the following functions:
   * - TransformPointUseAddition,
   * - TransformPointUseComposition,
   * - TransformPointNoCurrentTransform
   * - TransformPointNoInitialTransform.
   */
  TransformPointFunctionPointer m_SelectedTransformPointFunction;

  /**  A pointer to one of the following functions:
   * - GetJacobianUseAddition,
   * - GetJacobianUseComposition,
   * - GetJacobianNoCurrentTransform
   * - GetJacobianNoInitialTransform.
   */
  // GetJacobianFunctionPointer m_SelectedGetJacobianFunction;

  /** More of these. */
  GetSparseJacobianFunctionPointer                        m_SelectedGetSparseJacobianFunction;
  EvaluateJacobianWithImageGradientProductFunctionPointer m_SelectedEvaluateJacobianWithImageGradientProductFunction;
  GetSpatialJacobianFunctionPointer                       m_SelectedGetSpatialJacobianFunction;
  GetSpatialHessianFunctionPointer                        m_SelectedGetSpatialHessianFunction;
  GetJacobianOfSpatialJacobianFunctionPointer             m_SelectedGetJacobianOfSpatialJacobianFunction;
  GetJacobianOfSpatialJacobianFunctionPointer2            m_SelectedGetJacobianOfSpatialJacobianFunction2;
  GetJacobianOfSpatialHessianFunctionPointer              m_SelectedGetJacobianOfSpatialHessianFunction;
  GetJacobianOfSpatialHessianFunctionPointer2             m_SelectedGetJacobianOfSpatialHessianFunction2;

  /** How to combine the transformations. */
  bool m_UseAddition;
  bool m_UseComposition;

private:
  AdvancedCombinationTransform(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAdvancedCombinationTransform.hxx"
#endif

#endif // end #ifndef itkAdvancedCombinationTransform_h
