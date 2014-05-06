/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAdvancedVersorRigid3DTransform.h,v $
  Language:  C++
  Date:      $Date: 2006-08-09 04:35:32 $
  Version:   $Revision: 1.27 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkAdvancedVersorRigid3DTransform_h
#define __itkAdvancedVersorRigid3DTransform_h

#include <iostream>
#include "itkAdvancedVersorTransform.h"

namespace itk
{

/** \class AdvancedVersorRigid3DTransform
 *
 * \brief AdvancedVersorRigid3DTransform of a vector space (e.g. space coordinates)
 *
 * This transform applies a rotation and translation to the space
 * The parameters for this transform can be set either using individual Set
 * methods or in serialized form using SetParameters() and SetFixedParameters().
 *
 * The serialization of the optimizable parameters is an array of 6 elements.
 * The first 3 elements are the components of the versor representation
 * of 3D rotation. The last 3 parameters defines the translation in each
 * dimension.
 *
 * The serialization of the fixed parameters is an array of 3 elements defining
 * the center of rotation.
 *
 * NB: SK: this class is just to have the AdvancedSimilarity3DTransform. It is not complete.
 *
 * \ingroup Transforms
 */
template< class TScalarType = double >
//Data type for scalars (float or double)
class ITK_EXPORT AdvancedVersorRigid3DTransform :
  public         AdvancedVersorTransform< TScalarType >
{
public:

  /** Standard class typedefs. */
  typedef AdvancedVersorRigid3DTransform         Self;
  typedef AdvancedVersorTransform< TScalarType > Superclass;
  typedef SmartPointer< Self >                   Pointer;
  typedef SmartPointer< const Self >             ConstPointer;

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedVersorRigid3DTransform, AdvancedVersorTransform );

  /** Dimension of parameters. */
  itkStaticConstMacro( SpaceDimension, unsigned int, 3 );
  itkStaticConstMacro( InputSpaceDimension, unsigned int, 3 );
  itkStaticConstMacro( OutputSpaceDimension, unsigned int, 3 );
  itkStaticConstMacro( ParametersDimension, unsigned int, 6 );

  /** Parameters Type   */
  typedef typename Superclass::ParametersType         ParametersType;
  typedef typename Superclass::NumberOfParametersType NumberOfParametersType;
  typedef typename Superclass::JacobianType           JacobianType;
  typedef typename Superclass::ScalarType             ScalarType;
  typedef typename Superclass::InputPointType         InputPointType;
  typedef typename Superclass::OutputPointType        OutputPointType;
  typedef typename Superclass::InputVectorType        InputVectorType;
  typedef typename Superclass::OutputVectorType       OutputVectorType;
  typedef typename Superclass::InputVnlVectorType     InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType    OutputVnlVectorType;
  typedef typename Superclass::InputCovariantVectorType
    InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType
    OutputCovariantVectorType;
  typedef typename Superclass::MatrixType        MatrixType;
  typedef typename Superclass::InverseMatrixType InverseMatrixType;
  typedef typename Superclass::CenterType        CenterType;
  typedef typename Superclass::OffsetType        OffsetType;
  typedef typename Superclass::TranslationType   TranslationType;

  /** Versor type. */
  typedef typename Superclass::VersorType VersorType;
  typedef typename Superclass::AxisType   AxisType;
  typedef typename Superclass::AngleType  AngleType;

  typedef typename Superclass
    ::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType InternalMatrixType;

  /** Set the transformation from a container of parameters
   * This is typically used by optimizers.
   * There are 6 parameters. The first three represent the
   * versor, the last three represent the translation. */
  void SetParameters( const ParametersType & parameters );

  virtual const ParametersType & GetParameters( void ) const;

  /** This method computes the Jacobian matrix of the transformation. */
  virtual void GetJacobian(
    const InputPointType &,
    JacobianType &,
    NonZeroJacobianIndicesType & ) const;

protected:

  AdvancedVersorRigid3DTransform( unsigned int outputSpaceDim,
    unsigned int paramDim );
  AdvancedVersorRigid3DTransform( const MatrixType & matrix,
    const OutputVectorType & offset );
  AdvancedVersorRigid3DTransform();
  ~AdvancedVersorRigid3DTransform(){}

  void PrintSelf( std::ostream & os, Indent indent ) const;

  /** This method must be made protected here because it is not a safe way of
   * initializing the Versor */
  virtual void SetRotationMatrix( const MatrixType & matrix )
  { this->Superclass::SetRotationMatrix( matrix ); }

private:

  AdvancedVersorRigid3DTransform( const Self & ); //purposely not implemented
  void operator=( const Self & );                 //purposely not implemented

};

//class AdvancedVersorRigid3DTransform

}  // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedVersorRigid3DTransform.hxx"
#endif

#endif /* __itkAdvancedVersorRigid3DTransform_h */
