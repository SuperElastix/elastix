/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkAdvancedCombinationTransform_h
#define __itkAdvancedCombinationTransform_h

#include "itkAdvancedTransform.h"
#include "itkExceptionObject.h"

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

template< typename TScalarType, unsigned int NDimensions = 3 >
class AdvancedCombinationTransform :
  public AdvancedTransform< TScalarType, NDimensions, NDimensions >
{
public:

  /** Standard itk. */
  typedef AdvancedCombinationTransform Self;
  typedef AdvancedTransform< TScalarType,
    NDimensions, NDimensions >                Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** New method for creating an object using a factory. */
  itkNewMacro( Self );

  /** ITK Type info. */
  itkTypeMacro( AdvancedCombinationTransform, AdvancedTransform );

  /** Input and Output space dimension. */
  itkStaticConstMacro( SpaceDimension, unsigned int, NDimensions );

  /** Typedefs inherited from Superclass.*/
  typedef typename Superclass::ScalarType                    ScalarType;
  typedef typename Superclass::ParametersType                ParametersType;
  typedef typename Superclass::NumberOfParametersType        NumberOfParametersType;
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
  typedef typename Superclass::TransformCategoryType         TransformCategoryType;

  /** Transform typedefs for the from Superclass. */
  typedef typename Superclass::TransformType   TransformType;
  typedef typename TransformType::Pointer      TransformTypePointer;
  typedef typename TransformType::ConstPointer TransformTypeConstPointer;

  /** Typedefs for the InitialTransform. */
  typedef Superclass                                  InitialTransformType;
  typedef typename InitialTransformType::Pointer      InitialTransformPointer;
  typedef typename InitialTransformType::ConstPointer InitialTransformConstPointer;
  typedef typename InitialTransformType::InverseTransformBaseType
    InitialTransformInverseTransformBaseType;
  typedef typename InitialTransformType::InverseTransformBasePointer
    InitialTransformInverseTransformBasePointer;

  /** Typedefs for the CurrentTransform. */
  typedef Superclass                                  CurrentTransformType;
  typedef typename CurrentTransformType::Pointer      CurrentTransformPointer;
  typedef typename CurrentTransformType::ConstPointer CurrentTransformConstPointer;
  typedef typename CurrentTransformType::InverseTransformBaseType
    CurrentTransformInverseTransformBaseType;
  typedef typename CurrentTransformType::InverseTransformBasePointer
    CurrentTransformInverseTransformBasePointer;

  /** Set/Get a pointer to the InitialTransform. */
  virtual void SetInitialTransform( InitialTransformType * _arg );

  itkGetObjectMacro( InitialTransform, InitialTransformType );
  itkGetConstObjectMacro( InitialTransform, InitialTransformType );

  /** Set/Get a pointer to the CurrentTransform.
   * Make sure to set the CurrentTransform before calling functions like
   * TransformPoint(), GetJacobian(), SetParameters() etc.
   */
  virtual void SetCurrentTransform( CurrentTransformType * _arg );

  itkGetObjectMacro( CurrentTransform, CurrentTransformType );
  itkGetConstObjectMacro( CurrentTransform, CurrentTransformType );

  /** Return the number of sub-transforms. */
  virtual SizeValueType GetNumberOfTransforms( void ) const;

  /** Get the Nth current transform.
    * Exact interface to the ITK4 MultiTransform::GetNthTransform( SizeValueType n )
    * \warning The bounds checking is performed.
    */
  virtual const TransformTypePointer GetNthTransform( SizeValueType n ) const;

  /** Control the way transforms are combined. */
  virtual void SetUseComposition( bool _arg );

  itkGetConstMacro( UseComposition, bool );

  /** Control the way transforms are combined. */
  virtual void SetUseAddition( bool _arg );

  itkGetConstMacro( UseAddition, bool );

  /**  Method to transform a point. */
  virtual OutputPointType TransformPoint( const InputPointType  & point ) const;

  /** ITK4 change:
   * The following pure virtual functions must be overloaded.
   * For now just throw an exception, since these are not used in elastix.
   */
  virtual OutputVectorType TransformVector( const InputVectorType & ) const
  {
    itkExceptionMacro(
        << "TransformVector(const InputVectorType &) is not implemented "
        << "for AdvancedCombinationTransform" );
  }


  virtual OutputVnlVectorType TransformVector( const InputVnlVectorType & ) const
  {
    itkExceptionMacro(
        << "TransformVector(const InputVnlVectorType &) is not implemented "
        << "for AdvancedCombinationTransform" );
  }


  virtual OutputCovariantVectorType TransformCovariantVector( const InputCovariantVectorType & ) const
  {
    itkExceptionMacro(
        << "TransformCovariantVector(const InputCovariantVectorType &) is not implemented "
        << "for AdvancedCombinationTransform" );
  }


  /** Return the number of parameters that completely define the CurrentTransform. */
  virtual NumberOfParametersType GetNumberOfParameters( void ) const;

  /** Get the number of nonzero Jacobian indices. By default all. */
  virtual NumberOfParametersType GetNumberOfNonZeroJacobianIndices( void ) const;

  /** Get the transformation parameters from the CurrentTransform. */
  virtual const ParametersType & GetParameters( void ) const;

  /** Get the fixed parameters from the CurrentTransform. */
  virtual const ParametersType & GetFixedParameters( void ) const;

  /** Set the transformation parameters in the CurrentTransform. */
  virtual void SetParameters( const ParametersType & param );

  /** Set the transformation parameters in the CurrentTransform.
   * This method forces the transform to copy the parameters.
   */
  virtual void SetParametersByValue( const ParametersType & param );

  /** Set the fixed parameters in the CurrentTransform. */
  virtual void SetFixedParameters( const ParametersType & fixedParam );

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
  virtual bool GetInverse( Self * inverse ) const;

  /** Return whether the transform is linear (or actually: affine)
   * Returns true when both initial and current transform are linear */
  virtual bool IsLinear( void ) const;

  /** Special handling for combination transform. If all transforms
   * are linear, then return category Linear. Otherwise if all
   * transforms set to optimize are DisplacementFields, then
   * return DisplacementField category. */
  virtual TransformCategoryType GetTransformCategory() const;

  /** Whether the advanced transform has nonzero matrices. */
  virtual bool GetHasNonZeroSpatialHessian( void ) const;

  virtual bool HasNonZeroJacobianOfSpatialHessian( void ) const;

  /** Compute the (sparse) Jacobian of the transformation. */
  virtual void GetJacobian(
    const InputPointType & ipp,
    JacobianType & j,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Compute the spatial Jacobian of the transformation. */
  virtual void GetSpatialJacobian(
    const InputPointType & ipp,
    SpatialJacobianType & sj ) const;

  /** Compute the spatial Hessian of the transformation. */
  virtual void GetSpatialHessian(
    const InputPointType & ipp,
    SpatialHessianType & sh ) const;

  /** Compute the Jacobian of the spatial Jacobian of the transformation. */
  virtual void GetJacobianOfSpatialJacobian(
    const InputPointType & ipp,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Compute both the spatial Jacobian and the Jacobian of the
   * spatial Jacobian of the transformation.
   */
  virtual void GetJacobianOfSpatialJacobian(
    const InputPointType & ipp,
    SpatialJacobianType & sj,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Compute the Jacobian of the spatial Hessian of the transformation. */
  virtual void GetJacobianOfSpatialHessian(
    const InputPointType & ipp,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Compute both the spatial Hessian and the Jacobian of the
   * spatial Hessian of the transformation.
   */
  virtual void GetJacobianOfSpatialHessian(
    const InputPointType & ipp,
    SpatialHessianType & sh,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Typedefs for function pointers. */
  typedef OutputPointType (Self::* TransformPointFunctionPointer)( const InputPointType & ) const;
  typedef void (Self::*            GetSparseJacobianFunctionPointer)(
    const InputPointType &,
    JacobianType &,
    NonZeroJacobianIndicesType & ) const;
  typedef void (Self::* GetSpatialJacobianFunctionPointer)(
    const InputPointType &,
    SpatialJacobianType & ) const;
  typedef void (Self::* GetSpatialHessianFunctionPointer)(
    const InputPointType &,
    SpatialHessianType & ) const;
  typedef void (Self::* GetJacobianOfSpatialJacobianFunctionPointer)(
    const InputPointType &,
    JacobianOfSpatialJacobianType &,
    NonZeroJacobianIndicesType & ) const;
  typedef void (Self::* GetJacobianOfSpatialJacobianFunctionPointer2)(
    const InputPointType &,
    SpatialJacobianType &,
    JacobianOfSpatialJacobianType &,
    NonZeroJacobianIndicesType & ) const;
  typedef void (Self::* GetJacobianOfSpatialHessianFunctionPointer)(
    const InputPointType &,
    JacobianOfSpatialHessianType &,
    NonZeroJacobianIndicesType & ) const;
  typedef void (Self::* GetJacobianOfSpatialHessianFunctionPointer2)(
    const InputPointType &,
    SpatialHessianType &,
    JacobianOfSpatialHessianType &,
    NonZeroJacobianIndicesType & ) const;

protected:

  /** Constructor. */
  AdvancedCombinationTransform();

  /** Destructor. */
  virtual ~AdvancedCombinationTransform(){}

  /** Declaration of members. */
  InitialTransformPointer m_InitialTransform;
  CurrentTransformPointer m_CurrentTransform;

  /** Set the SelectedTransformPointFunction and the
   * SelectedGetJacobianFunction.
   */
  virtual void UpdateCombinationMethod( void );

  /** Throw an exception. */
  virtual void NoCurrentTransformSet( void ) const throw ( ExceptionObject );

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
  //GetJacobianFunctionPointer m_SelectedGetJacobianFunction;

  /** More of these. */
  GetSparseJacobianFunctionPointer             m_SelectedGetSparseJacobianFunction;
  GetSpatialJacobianFunctionPointer            m_SelectedGetSpatialJacobianFunction;
  GetSpatialHessianFunctionPointer             m_SelectedGetSpatialHessianFunction;
  GetJacobianOfSpatialJacobianFunctionPointer  m_SelectedGetJacobianOfSpatialJacobianFunction;
  GetJacobianOfSpatialJacobianFunctionPointer2 m_SelectedGetJacobianOfSpatialJacobianFunction2;
  GetJacobianOfSpatialHessianFunctionPointer   m_SelectedGetJacobianOfSpatialHessianFunction;
  GetJacobianOfSpatialHessianFunctionPointer2  m_SelectedGetJacobianOfSpatialHessianFunction2;

  /** ************************************************
   * Methods to transform a point.
   */

  /** ADDITION: \f$T(x) = T_0(x) + T_1(x) - x\f$ */
  inline OutputPointType TransformPointUseAddition(
    const InputPointType & point ) const;

  /** COMPOSITION: \f$T(x) = T_1( T_0(x) )\f$
   * \warning: assumes that input and output point type are the same.
   */
  inline OutputPointType TransformPointUseComposition(
    const InputPointType & point ) const;

  /** CURRENT ONLY: \f$T(x) = T_1(x)\f$ */
  inline OutputPointType TransformPointNoInitialTransform(
    const InputPointType & point ) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline OutputPointType TransformPointNoCurrentTransform(
    const InputPointType & point ) const;

  /** ************************************************
   * Methods to compute the sparse Jacobian.
   */

  /** ADDITION: \f$J(x) = J_1(x)\f$ */
  inline void GetJacobianUseAddition(
    const InputPointType &,
    JacobianType &,
    NonZeroJacobianIndicesType & ) const;

  /** COMPOSITION: \f$J(x) = J_1( T_0(x) )\f$
   * \warning: assumes that input and output point type are the same.
   */
  inline void GetJacobianUseComposition(
    const InputPointType &,
    JacobianType &,
    NonZeroJacobianIndicesType & ) const;

  /** CURRENT ONLY: \f$J(x) = J_1(x)\f$ */
  inline void GetJacobianNoInitialTransform(
    const InputPointType &,
    JacobianType &,
    NonZeroJacobianIndicesType & ) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline void GetJacobianNoCurrentTransform(
    const InputPointType &,
    JacobianType &,
    NonZeroJacobianIndicesType & ) const;

  /** ************************************************
   * Methods to compute the spatial Jacobian.
   */

  /** ADDITION: \f$J(x) = J_1(x)\f$ */
  inline void GetSpatialJacobianUseAddition(
    const InputPointType & ipp,
    SpatialJacobianType & sj ) const;

  /** COMPOSITION: \f$J(x) = J_1( T_0(x) )\f$
   * \warning: assumes that input and output point type are the same.
   */
  inline void GetSpatialJacobianUseComposition(
    const InputPointType & ipp,
    SpatialJacobianType & sj ) const;

  /** CURRENT ONLY: \f$J(x) = J_1(x)\f$ */
  inline void GetSpatialJacobianNoInitialTransform(
    const InputPointType & ipp,
    SpatialJacobianType & sj ) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline void GetSpatialJacobianNoCurrentTransform(
    const InputPointType & ipp,
    SpatialJacobianType & sj ) const;

  /** ************************************************
   * Methods to compute the spatial Hessian.
   */

  /** ADDITION: \f$J(x) = J_1(x)\f$ */
  inline void GetSpatialHessianUseAddition(
    const InputPointType & ipp,
    SpatialHessianType & sh ) const;

  /** COMPOSITION: \f$J(x) = J_1( T_0(x) )\f$
   * \warning: assumes that input and output point type are the same.
   */
  inline void GetSpatialHessianUseComposition(
    const InputPointType & ipp,
    SpatialHessianType & sh ) const;

  /** CURRENT ONLY: \f$J(x) = J_1(x)\f$ */
  inline void GetSpatialHessianNoInitialTransform(
    const InputPointType & ipp,
    SpatialHessianType & sh ) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline void GetSpatialHessianNoCurrentTransform(
    const InputPointType & ipp,
    SpatialHessianType & sh ) const;

  /** ************************************************
   * Methods to compute the Jacobian of the spatial Jacobian.
   */

  /** ADDITION: \f$J(x) = J_1(x)\f$ */
  inline void GetJacobianOfSpatialJacobianUseAddition(
    const InputPointType & ipp,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  inline void GetJacobianOfSpatialJacobianUseAddition(
    const InputPointType & ipp,
    SpatialJacobianType & sj,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** COMPOSITION: \f$J(x) = J_1( T_0(x) )\f$
   * \warning: assumes that input and output point type are the same.
   */
  inline void GetJacobianOfSpatialJacobianUseComposition(
    const InputPointType & ipp,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  inline void GetJacobianOfSpatialJacobianUseComposition(
    const InputPointType & ipp,
    SpatialJacobianType & sj,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** CURRENT ONLY: \f$J(x) = J_1(x)\f$ */
  inline void GetJacobianOfSpatialJacobianNoInitialTransform(
    const InputPointType & ipp,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  inline void GetJacobianOfSpatialJacobianNoInitialTransform(
    const InputPointType & ipp,
    SpatialJacobianType & sj,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline void GetJacobianOfSpatialJacobianNoCurrentTransform(
    const InputPointType & ipp,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  inline void GetJacobianOfSpatialJacobianNoCurrentTransform(
    const InputPointType & ipp,
    SpatialJacobianType & sj,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** ************************************************
   * Methods to compute the Jacobian of the spatial Hessian.
   */

  /** ADDITION: \f$J(x) = J_1(x)\f$ */
  inline void GetJacobianOfSpatialHessianUseAddition(
    const InputPointType & ipp,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  inline void GetJacobianOfSpatialHessianUseAddition(
    const InputPointType & ipp,
    SpatialHessianType & sh,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** COMPOSITION: \f$J(x) = J_1( T_0(x) )\f$
   * \warning: assumes that input and output point type are the same.
   */
  inline void GetJacobianOfSpatialHessianUseComposition(
    const InputPointType & ipp,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  virtual inline void GetJacobianOfSpatialHessianUseComposition(
    const InputPointType & ipp,
    SpatialHessianType & sh,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** CURRENT ONLY: \f$J(x) = J_1(x)\f$ */
  inline void GetJacobianOfSpatialHessianNoInitialTransform(
    const InputPointType & ipp,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  inline void GetJacobianOfSpatialHessianNoInitialTransform(
    const InputPointType & ipp,
    SpatialHessianType & sh,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline void GetJacobianOfSpatialHessianNoCurrentTransform(
    const InputPointType & ipp,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  inline void GetJacobianOfSpatialHessianNoCurrentTransform(
    const InputPointType & ipp,
    SpatialHessianType & sh,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** How to combine the transformations. */
  bool m_UseAddition;
  bool m_UseComposition;

private:

  AdvancedCombinationTransform( const Self & ); // purposely not implemented
  void operator=( const Self & );               // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedCombinationTransform.hxx"
#endif

#endif // end #ifndef __itkAdvancedCombinationTransform_h
