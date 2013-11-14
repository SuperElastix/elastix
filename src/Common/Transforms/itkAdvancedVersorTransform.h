/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAdvancedVersorTransform.h,v $
  Language:  C++
  Date:      $Date: 2006-08-09 04:35:32 $
  Version:   $Revision: 1.17 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#ifndef __itkAdvancedVersorTransform_h
#define __itkAdvancedVersorTransform_h

#include <iostream>
#include "itkAdvancedRigid3DTransform.h"
#include "vnl/vnl_quaternion.h"
#include "itkVersor.h"

namespace itk
{

/**
 *
 * AdvancedVersorTransform of a vector space (e.g. space coordinates)
 *
 * This transform applies a rotation to the space. Rotation is about
 * a user specified center.
 *
 * The serialization of the optimizable parameters is an array of 3 elements
 * representing the right part of the versor.
 *
 * The serialization of the fixed parameters is an array of 3 elements defining
 * the center of rotation.
 *
 * \todo Need to make sure that the translation parameters in the baseclass
 * cannot be set to non-zero values.
 *
 * NB: SK: this class is just to have the AdvancedSimilarity3DTransform. It is not complete.
 *
 * \ingroup Transforms
 *
 **/
template < class TScalarType=double >//Data type for scalars (float or double)
class ITK_EXPORT AdvancedVersorTransform : public AdvancedRigid3DTransform< TScalarType >
{
public:

  /** Standard Self Typedef */
  typedef AdvancedVersorTransform                   Self;
  typedef AdvancedRigid3DTransform< TScalarType >   Superclass;
  typedef SmartPointer<Self>                Pointer;
  typedef SmartPointer<const Self>          ConstPointer;

  /** Run-time type information (and related methods).  */
  itkTypeMacro( AdvancedVersorTransform, AdvancedRigid3DTransform );

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro( Self );

  /** Dimension of parameters */
  itkStaticConstMacro(SpaceDimension, unsigned int, 3);
  itkStaticConstMacro(InputSpaceDimension, unsigned int, 3);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, 3);
  itkStaticConstMacro(ParametersDimension, unsigned int, 3);

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
  typedef typename Superclass::MatrixType             MatrixType;
  typedef typename Superclass::InverseMatrixType      InverseMatrixType;
  typedef typename Superclass::CenterType             CenterType;
  typedef typename Superclass::OffsetType             OffsetType;

  typedef typename Superclass
    ::NonZeroJacobianIndicesType                    NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType  SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType                 JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType   SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType                  JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType   InternalMatrixType;

  /** VnlQuaternion Type */
  typedef vnl_quaternion<TScalarType>                 VnlQuaternionType;

  /** Versor Type */
  typedef Versor<TScalarType>                   VersorType;
  typedef typename VersorType::VectorType       AxisType;
  typedef typename VersorType::ValueType        AngleType;

  /**
   * Set the transformation from a container of parameters
   * This is typically used by optimizers.
   *
   * There are 3 parameters. They represent the components
   * of the right part of the versor. This can be seen
   * as the components of the vector parallel to the rotation
   * axis and multiplied by vcl_sin( angle / 2 ). */
  void SetParameters( const ParametersType & parameters );

  /** Get the Transformation Parameters. */
  const ParametersType& GetParameters(void) const;

  /** Set the rotational part of the transform */
  void SetRotation( const VersorType & versor );
  void SetRotation( const AxisType & axis, AngleType angle );
  itkGetConstReferenceMacro(Versor, VersorType);

  /** Set the parameters to the IdentityTransform */
  virtual void SetIdentity(void);

  /** This method computes the Jacobian matrix of the transformation. */
  virtual void GetJacobian(
    const InputPointType &,
    JacobianType &,
    NonZeroJacobianIndicesType & ) const;

protected:

  /** Construct an AdvancedVersorTransform object */
  AdvancedVersorTransform(const MatrixType &matrix,
                  const OutputVectorType &offset);
  AdvancedVersorTransform(unsigned int paramDims);
  AdvancedVersorTransform();

  /** Destroy an AdvancedVersorTransform object */
  ~AdvancedVersorTransform(){};

  /** This method must be made protected here because it is not a safe way of
   * initializing the Versor */
  virtual void SetRotationMatrix(const MatrixType & matrix)
    { this->Superclass::SetRotationMatrix( matrix ); }

  void SetVarVersor(const VersorType & newVersor)
    { m_Versor = newVersor; }

  /** Print contents of a AdvancedVersorTransform */
  void PrintSelf(std::ostream &os, Indent indent) const;

  /** Compute Matrix
   *  Compute the components of the rotation matrix in the superclass */
  void ComputeMatrix(void);
  void ComputeMatrixParameters(void);

private:
  /** Copy a AdvancedVersorTransform object */
  AdvancedVersorTransform(const Self & other); // Not implemented

  /** Assignment operator */
  const Self & operator=( const Self & ); // Not implemented

  /** Versor containing the rotation */
  VersorType    m_Versor;

}; //class AdvancedVersorTransform


}  // namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedVersorTransform.txx"
#endif

#endif /* __itkAdvancedVersorTransform_h */
