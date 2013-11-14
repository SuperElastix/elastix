/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAdvancedSimilarity3DTransform.h,v $
  Language:  C++
  Date:      $Date: 2006-08-09 04:35:32 $
  Version:   $Revision: 1.3 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#ifndef __itkAdvancedSimilarity3DTransform_h
#define __itkAdvancedSimilarity3DTransform_h

#include <iostream>
#include "itkAdvancedVersorRigid3DTransform.h"

namespace itk
{

/** \brief AdvancedSimilarity3DTransform of a vector space (e.g. space coordinates)
 *
 * This transform applies a rotation, translation and isotropic scaling to the space.
 *
 * The parameters for this transform can be set either using individual Set
 * methods or in serialized form using SetParameters() and SetFixedParameters().
 *
 * The serialization of the optimizable parameters is an array of 7 elements.
 * The first 3 elements are the components of the versor representation
 * of 3D rotation. The next 3 parameters defines the translation in each
 * dimension. The last parameter defines the isotropic scaling.
 *
 * The serialization of the fixed parameters is an array of 3 elements defining
 * the center of rotation.
 *
 * \ingroup Transforms
 *
 * \sa VersorRigid3DTransform
 */
template < class TScalarType=double >    // Data type for scalars (float or double)
class ITK_EXPORT AdvancedSimilarity3DTransform :
      public AdvancedVersorRigid3DTransform< TScalarType >
{
public:
  /** Standard class typedefs. */
  typedef AdvancedSimilarity3DTransform                  Self;
  typedef AdvancedVersorRigid3DTransform< TScalarType >  Superclass;
  typedef SmartPointer<Self>                     Pointer;
  typedef SmartPointer<const Self>               ConstPointer;

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedSimilarity3DTransform, AdvancedVersorRigid3DTransform );

  /** Dimension of parameters. */
  itkStaticConstMacro(SpaceDimension, unsigned int, 3);
  itkStaticConstMacro(InputSpaceDimension, unsigned int, 3);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, 3);
  itkStaticConstMacro(ParametersDimension, unsigned int, 7);

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
  typedef typename Superclass::TranslationType        TranslationType;

  /** Versor type. */
  typedef typename Superclass::VersorType             VersorType;
  typedef typename Superclass::AxisType               AxisType;
  typedef typename Superclass::AngleType              AngleType;
  typedef          TScalarType                        ScaleType;

  typedef typename Superclass
    ::NonZeroJacobianIndicesType                    NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType  SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType                 JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType   SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType                  JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType   InternalMatrixType;

 /** Directly set the rotation matrix of the transform.
  * \warning The input matrix must be orthogonal with isotropic scaling
  * to within a specified tolerance, else an exception is thrown.
  *
  * \sa MatrixOffsetTransformBase::SetMatrix() */
  virtual void SetMatrix(const MatrixType &matrix);

  /** Set the transformation from a container of parameters This is typically
   * used by optimizers.  There are 7 parameters. The first three represent the
   * versor, the next three represent the translation and the last one
   * represents the scaling factor. */
  void SetParameters( const ParametersType & parameters );
  virtual const ParametersType& GetParameters(void) const;

  /** Set/Get the value of the isotropic scaling factor */
  void SetScale( ScaleType scale );
  itkGetConstReferenceMacro( Scale, ScaleType );

  /** This method computes the Jacobian matrix of the transformation. */
  virtual void GetJacobian(
    const InputPointType &,
    JacobianType &,
    NonZeroJacobianIndicesType & ) const;

protected:
  AdvancedSimilarity3DTransform(unsigned int outputSpaceDim,
                         unsigned int paramDim);
  AdvancedSimilarity3DTransform(const MatrixType & matrix,
                         const OutputVectorType & offset);
  AdvancedSimilarity3DTransform();
  ~AdvancedSimilarity3DTransform(){};

  void PrintSelf(std::ostream &os, Indent indent) const;

  /** Recomputes the matrix by calling the Superclass::ComputeMatrix() and then
   * applying the scale factor. */
  void ComputeMatrix();

  /** Computes the parameters from an input matrix. */
  void ComputeMatrixParameters();

   /** Update the m_JacobianOfSpatialJacobian.  */
  virtual void PrecomputeJacobianOfSpatialJacobian(void);

private:
  AdvancedSimilarity3DTransform(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  ScaleType    m_Scale;

}; //class AdvancedSimilarity3DTransform


}  // namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedSimilarity3DTransform.txx"
#endif

#endif /* __itkAdvancedSimilarity3DTransform_h */
