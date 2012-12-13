/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkAffineLogTransform_h
#define __itkAffineLogTransform_h

#include <iostream>
#include "itkAdvancedMatrixOffsetTransformBase.h"

namespace itk
{

/** \class AffineLogTransform
 *
 *
 * \ingroup Transforms
 */
template < class TScalarType=double, unsigned int Dimension=2 >    // Data type for scalars (float or double)
class AffineLogTransform:
     public AdvancedMatrixOffsetTransformBase< TScalarType, Dimension, Dimension >
{
public:
  /** Standard class typedefs. */
  typedef AffineLogTransform                  Self;
  typedef AdvancedMatrixOffsetTransformBase< TScalarType, Dimension, Dimension >   Superclass;
  typedef SmartPointer<Self>                Pointer;
  typedef SmartPointer<const Self>          ConstPointer;

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AffineLogTransform, AdvancedMatrixOffsetTransformBase );

  /** Dimension of the domain space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, Dimension);
  itkStaticConstMacro( OutputSpaceDimension, unsigned int, Dimension);
  itkStaticConstMacro( InputSpaceDimension, unsigned int, Dimension);
	itkStaticConstMacro( ParametersDimension, unsigned int, (Dimension+1)*Dimension);

  typedef typename Superclass::ParametersType             ParametersType;
  typedef typename Superclass::NumberOfParametersType     NumberOfParametersType;
  typedef typename Superclass::JacobianType               JacobianType;
  typedef typename Superclass::ScalarType                 ScalarType;
  typedef typename Superclass::InputVectorType            InputVectorType;
  typedef typename Superclass::OutputVectorType           OutputVectorType;
  typedef typename Superclass::InputCovariantVectorType   InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType  OutputCovariantVectorType;
  typedef typename Superclass::InputVnlVectorType         InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType        OutputVnlVectorType;
  typedef typename Superclass::InputPointType             InputPointType;
  typedef typename Superclass::OutputPointType            OutputPointType;
  typedef typename Superclass::MatrixType                 MatrixType;
  typedef typename Superclass::InverseMatrixType          InverseMatrixType;
  typedef typename Superclass::CenterType                 CenterType;
  typedef typename Superclass::TranslationType            TranslationType;
  typedef typename Superclass::OffsetType                 OffsetType;
  typedef typename Superclass::ScalarType                 AngleType;

  typedef typename Superclass
    ::NonZeroJacobianIndicesType                    NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType  SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType                 JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType   SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType                  JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType   InternalMatrixType;

  typedef FixedArray< ScalarType >                  ScalarArrayType;

  void SetParameters( const ParametersType & parameters );
  const ParametersType & GetParameters(void) const;

  /** Compute the Jacobian of the transformation. */
  virtual void GetJacobian(
    const InputPointType &,
    JacobianType &,
    NonZeroJacobianIndicesType & ) const;

  virtual void SetIdentity(void);

protected:
  AffineLogTransform();
  AffineLogTransform(const MatrixType & matrix,
                   const OutputPointType & offset);

  ~AffineLogTransform(){};

  void PrintSelf(std::ostream &os, Indent indent) const;

  /** Compute the components of the rotation matrix in the superclass. */
  void ComputeMatrixLogDomain( void );

   /** Update the m_JacobianOfSpatialJacobian.  */
  virtual void PrecomputeJacobianOfSpatialJacobian(void);

private:
  AffineLogTransform(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  vnl_matrix<ScalarType> m_MatrixLogDomain;
  vnl_matrix<ScalarType> m_Matrix;
  OffsetType m_Offset;


}; //class AffineLogTransform


}  // namespace itk
#if ITK_TEMPLATE_EXPLICIT
# include "Templates/itkAffineLogTransform+-.h"
#endif

#if ITK_TEMPLATE_TXX
# include "itkAffineLogTransform.txx"
#endif

#endif /* __itkAffineLogTransform_h */
