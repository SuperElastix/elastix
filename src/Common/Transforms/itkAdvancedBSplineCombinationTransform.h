/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkAdvancedBSplineCombinationTransform_h
#define __itkAdvancedBSplineCombinationTransform_h

#include "itkAdvancedCombinationTransform.h"
#include "itkAdvancedBSplineDeformableTransform.h"


namespace itk
{
  
/**
 * \class AdvancedBSplineCombinationTransform
 *
 * \brief This class combines two transforms: an 'initial transform'
 * with a 'current transform'.
 *
 * The itk::BSplineCombinationTransform class combines an initial transform \f$T_0\f$ with a
 * current transform \f$T_1\f$. The current transform is expected to be 
 * a itk::BSplineDeformableTransform. This extra class, specific for BSplineTransforms
 * is necessary because the itk::BSplineDeformableTransform has an extra 
 * TransformPoint() method, with extra arguments, used by the 
 * itk::MattesMutualInformationImageToImageMetric to speed up registration.
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
 * So, the transform parameters of the BSplineCombinationTransform are the
 * parameters of the CurrentTransform T1.
 *
 * Note: It is mandatory to set a current transform. An initial transform
 * is not mandatory. Make sure that the current transform is a 
 * itk::BSplineDeformableTransform.
 * 
 * \ingroup Transforms
 */

template < typename TScalarType, unsigned int NDimensions = 3, unsigned int VSplineOrder = 3 >
class AdvancedBSplineCombinationTransform
  : public AdvancedCombinationTransform<TScalarType, NDimensions>
{
public:

  /** Standard ITK. */
  typedef AdvancedBSplineCombinationTransform Self;
  typedef AdvancedCombinationTransform<
    TScalarType,
    NDimensions >                     Superclass;
  typedef SmartPointer< Self >        Pointer;
  typedef SmartPointer< const Self >  ConstPointer;

  /** New method for creating an object using a factory. */
  itkNewMacro( Self );

  /** ITK type info. */
  itkTypeMacro( AdvancedBSplineCombinationTransform, AdvancedCombinationTransform );

  /** Input and Output space dimension. */
  itkStaticConstMacro( SpaceDimension, unsigned int, NDimensions );

  /** BSplineOrder. */
  itkStaticConstMacro( SplineOrder, unsigned int, VSplineOrder );

  /** Typedefs inherited from Superclass. */     
  typedef typename Superclass::ScalarType                   ScalarType;
  typedef typename Superclass::ParametersType               ParametersType;
  typedef typename Superclass::JacobianType                 JacobianType;
  typedef typename Superclass::InputVectorType              InputVectorType;
  typedef typename Superclass::OutputVectorType             OutputVectorType;
  typedef typename Superclass::InputCovariantVectorType     InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType    OutputCovariantVectorType;
  typedef typename Superclass::InputVnlVectorType           InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType          OutputVnlVectorType;
  typedef typename Superclass::InputPointType               InputPointType;
  typedef typename Superclass::OutputPointType              OutputPointType;
  typedef typename 
    Superclass::TransformPointFunctionPointer               TransformPointFunctionPointer;
  typedef typename 
    Superclass::GetJacobianFunctionPointer                  GetJacobianFunctionPointer;
  typedef typename Superclass::InitialTransformType         InitialTransformType;
  typedef typename Superclass::InitialTransformPointer      InitialTransformPointer;
  typedef typename Superclass::InitialTransformConstPointer InitialTransformConstPointer;
  typedef typename Superclass::CurrentTransformType         CurrentTransformType;
  typedef typename Superclass::CurrentTransformPointer      CurrentTransformPointer;
  typedef typename Superclass::NonZeroJacobianIndicesType     NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType            SpatialJacobianType;
  typedef typename Superclass::JacobianOfSpatialJacobianType  JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType             SpatialHessianType;
  typedef typename Superclass::JacobianOfSpatialHessianType   JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType             InternalMatrixType;

  /** Typedefs for the AdvancedBSplineTransform. */
  typedef itk::AdvancedBSplineDeformableTransform<
    ScalarType,
    itkGetStaticConstMacro( SpaceDimension ),
    itkGetStaticConstMacro( SplineOrder ) >                 BSplineTransformType;
  typedef typename BSplineTransformType::Pointer            BSplineTransformPointer;
  typedef typename BSplineTransformType::WeightsType        WeightsType;
  typedef typename 
    BSplineTransformType::ParameterIndexArrayType           ParameterIndexArrayType;
  typedef typename BSplineTransformType::ImageType          CoefficientImageType;
  typedef typename BSplineTransformType::ImagePointer       CoefficientImagePointer;

  /** A pointer to a function that looks like the TransformPoint function 
   * with 5 arguments, as defined in the BSplineTransform.
   */
  typedef void (Self::*TransformPointBSplineFunctionPointer)(
    const InputPointType &, OutputPointType &, WeightsType &,
    ParameterIndexArrayType &, bool &) const;

  /** Set/Get a pointer to the CurrentTransform. Make sure to set
   * the CurrentTransform before functions like TransformPoint(),
   * GetJacobian(), SetParameters() etc. are called.
   * 
   * Tries to cast the given pointer to a B-spline transform. If
   * this is not possible, an exception error is the consequence.
   */
  virtual void SetCurrentTransform( CurrentTransformType * _arg );

  /**  Method to transform a point. This method just calls the 
   * superclass' implementation, but has to be present here, to avoid
   * compilation errors (because of the overloaded TransformPoint() method
   * that's also present in this class.
   */
  virtual OutputPointType TransformPoint( const InputPointType  & point ) const;

  /**  Method to transform a point with extra arguments. */
  virtual void TransformPoint(
    const InputPointType & inputPoint,
    OutputPointType & outputPoint,
    WeightsType & weights,
    ParameterIndexArrayType & indices, 
    bool & inside ) const;

protected:

  AdvancedBSplineCombinationTransform();
  virtual ~AdvancedBSplineCombinationTransform(){};

  /** The current transform casted to an AdvancedBSplineTransform. */
  BSplineTransformPointer m_CurrentTransformAsBSplineTransform;

  /** Set, besides the SelectedTransformPointFunction and the 
   * SelectedGetJacobianFunction, the SelectedTransformPointBSplineFunction.
   */
  virtual void UpdateCombinationMethod( void );

  /**  A pointer to one of the following functions:
   * - TransformPointBSplineUseAddition,
   * - TransformPointBSplineUseComposition,
   * - TransformPointBSplineNoCurrentTransform
   * - TransformPointBSplineNoInitialTransform. 
   */
  TransformPointBSplineFunctionPointer m_SelectedTransformPointBSplineFunction;

  /** Methods to combine the TransformPoint functions of the 
   * initial and the current transform. Variant with extra arguments.
   */

  /** ADDITION: T(x) = T0(x) + T1(x) - x. */
  inline void TransformPointBSplineUseAddition( 
    const InputPointType & inputPoint,
    OutputPointType & outputPoint,
    WeightsType & weights,
    ParameterIndexArrayType & indices, 
    bool & inside ) const;

  /** COMPOSITION: T(x) = T1( T0(x) ) 
   * \warning: assumes that input and output point type are the same.
   */
  inline void TransformPointBSplineUseComposition(
    const InputPointType & inputPoint,
    OutputPointType & outputPoint,
    WeightsType & weights,
    ParameterIndexArrayType & indices, 
    bool & inside ) const;

  /** CURRENT ONLY: T(x) = T1(x). */
  inline void TransformPointBSplineNoInitialTransform( 
    const InputPointType & inputPoint,
    OutputPointType & outputPoint,
    WeightsType & weights,
    ParameterIndexArrayType & indices, 
    bool & inside ) const;

  /** NO CURRENT TRANSFORM SET: throw an exception. */
  inline void TransformPointBSplineNoCurrentTransform( 
    const InputPointType & inputPoint,
    OutputPointType & outputPoint,
    WeightsType & weights,
    ParameterIndexArrayType & indices, 
    bool & inside ) const;

private:

  AdvancedBSplineCombinationTransform( const Self& ); // purposely not implemented
  void operator=( const Self& );    // purposely not implemented

}; // end class AdvancedBSplineCombinationTransform


} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedBSplineCombinationTransform.hxx"
#endif


#endif // end #ifndef __itkAdvancedBSplineCombinationTransform_h

