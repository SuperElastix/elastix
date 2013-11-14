/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/

/*

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAdvancedMatrixOffsetTransformBase.h,v $
  Language:  C++
  Date:      $Date: 2008-06-29 12:58:58 $
  Version:   $Revision: 1.20 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#ifndef __itkAdvancedMatrixOffsetTransformBase_h
#define __itkAdvancedMatrixOffsetTransformBase_h

#include <iostream>

#include "itkMatrix.h"
#include "itkAdvancedTransform.h"
#include "itkExceptionObject.h"
#include "itkMacro.h"

namespace itk
{


/**
 * Matrix and Offset transformation of a vector space (e.g. space coordinates)
 *
 * This class serves as a base class for transforms that can be expressed
 * as a linear transformation plus a constant offset (e.g., affine, similarity
 * and rigid transforms).   This base class also provides the concept of
 * using a center of rotation and a translation instead of an offset.
 *
 * As derived instances of this class are specializations of an affine
 * transform, any two of these transformations may be composed and the result
 * is an affine transformation.  However, the order is important.
 * Given two affine transformations T1 and T2, we will say that
 * "precomposing T1 with T2" yields the transformation which applies
 * T1 to the source, and then applies T2 to that result to obtain the
 * target.  Conversely, we will say that "postcomposing T1 with T2"
 * yields the transformation which applies T2 to the source, and then
 * applies T1 to that result to obtain the target.  (Whether T1 or T2
 * comes first lexicographically depends on whether you choose to
 * write mappings from right-to-left or vice versa; we avoid the whole
 * problem by referring to the order of application rather than the
 * textual order.)
 *
 * There are three template parameters for this class:
 *
 * ScalarT       The type to be used for scalar numeric values.  Either
 *               float or double.
 *
 * NInputDimensions   The number of dimensions of the input vector space.
 *
 * NOutputDimensions   The number of dimensions of the output vector space.
 *
 * This class provides several methods for setting the matrix and offset
 * defining the transform. To support the registration framework, the
 * transform parameters can also be set as an Array<double> of size
 * (NInputDimension + 1) * NOutputDimension using method SetParameters().
 * The first (NOutputDimension x NInputDimension) parameters defines the
 * matrix in row-major order (where the column index varies the fastest).
 * The last NOutputDimension parameters defines the translation
 * in each dimensions.
 *
 * \ingroup Transforms
 *
 */

template <
  class TScalarType=double,         // Data type for scalars
  unsigned int NInputDimensions=3,  // Number of dimensions in the input space
  unsigned int NOutputDimensions=3> // Number of dimensions in the output space
class AdvancedMatrixOffsetTransformBase
  : public AdvancedTransform< TScalarType, NInputDimensions, NOutputDimensions >
{
public:
  /** Standard typedefs   */
  typedef AdvancedMatrixOffsetTransformBase     Self;
  typedef AdvancedTransform< TScalarType,
    NInputDimensions, NOutputDimensions >       Superclass;
  typedef SmartPointer<Self>                    Pointer;
  typedef SmartPointer<const Self>              ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedMatrixOffsetTransformBase, AdvancedTransform );

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro( Self );

  /** Dimension of the domain space. */
  itkStaticConstMacro( InputSpaceDimension, unsigned int, NInputDimensions );
  itkStaticConstMacro( OutputSpaceDimension, unsigned int, NOutputDimensions );
  itkStaticConstMacro( ParametersDimension, unsigned int,
    NOutputDimensions * ( NInputDimensions + 1 ) );

  /** Typedefs from the Superclass. */
  typedef typename Superclass::ScalarType           ScalarType;
  typedef typename Superclass::ParametersType       ParametersType;
  typedef typename Superclass::NumberOfParametersType NumberOfParametersType;
  typedef typename Superclass::JacobianType         JacobianType;
  typedef typename Superclass::InputVectorType      InputVectorType;
  typedef typename Superclass::OutputVectorType     OutputVectorType;
  typedef typename Superclass
    ::InputCovariantVectorType                      InputCovariantVectorType;
  typedef typename Superclass
    ::OutputCovariantVectorType                     OutputCovariantVectorType;
  typedef typename Superclass::InputVnlVectorType   InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType  OutputVnlVectorType;
  typedef typename Superclass::InputPointType       InputPointType;
  typedef typename Superclass::OutputPointType      OutputPointType;
  typedef typename Superclass::TransformCategoryType TransformCategoryType;

  typedef typename Superclass
    ::NonZeroJacobianIndicesType                    NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType  SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType                 JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType   SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType                  JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType   InternalMatrixType;

  /** Standard matrix type for this class. */
  typedef Matrix< TScalarType,
    itkGetStaticConstMacro( OutputSpaceDimension ),
    itkGetStaticConstMacro( InputSpaceDimension )>  MatrixType;

  /** Standard inverse matrix type for this class. */
  typedef Matrix< TScalarType,
    itkGetStaticConstMacro( InputSpaceDimension ),
    itkGetStaticConstMacro( OutputSpaceDimension )> InverseMatrixType;

  /** Typedefs. */
  typedef InputPointType                            CenterType;
  typedef OutputVectorType                          OffsetType;
  typedef OutputVectorType                          TranslationType;

  /** Set the transformation to an Identity
   * This sets the matrix to identity and the Offset to null.
   */
  virtual void SetIdentity( void );

  /** Set matrix of an AdvancedMatrixOffsetTransformBase
   *
   * This method sets the matrix of an AdvancedMatrixOffsetTransformBase to a
   * value specified by the user.
   *
   * This updates the Offset wrt to current translation
   * and center.  See the warning regarding offset-versus-translation
   * in the documentation for SetCenter.
   *
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset.
   */
  virtual void SetMatrix( const MatrixType & matrix )
  {
    this->m_Matrix = matrix;
    this->ComputeOffset();
    this->ComputeMatrixParameters();
    this->m_MatrixMTime.Modified();
    this->Modified();
  }

  /** Get matrix of an AdvancedMatrixOffsetTransformBase
   *
   * This method returns the value of the matrix of the
   * AdvancedMatrixOffsetTransformBase.
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset.
   */
  const MatrixType & GetMatrix( void ) const
  {
    return this->m_Matrix;
  }

  /** Set offset (origin) of an MatrixOffset TransformBase.
   *
   * This method sets the offset of an AdvancedMatrixOffsetTransformBase to a
   * value specified by the user.
   * This updates Translation wrt current center.  See the warning regarding
   * offset-versus-translation in the documentation for SetCenter.
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset.
   */
  void SetOffset( const OutputVectorType &offset )
  {
    this->m_Offset = offset;
    this->ComputeTranslation();
    this->Modified();
  }

  /** Get offset of an AdvancedMatrixOffsetTransformBase
   *
   * This method returns the offset value of the AdvancedMatrixOffsetTransformBase.
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset.
   */
  const OutputVectorType & GetOffset( void ) const
  {
    return this->m_Offset;
  }

  /** Set center of rotation of an AdvancedMatrixOffsetTransformBase
   *
   * This method sets the center of rotation of an AdvancedMatrixOffsetTransformBase
   * to a fixed point - for most transforms derived from this class,
   * this point is not a "parameter" of the transform - the exception is that
   * "centered" transforms have center as a parameter during optimization.
   *
   * This method updates offset wrt to current translation and matrix.
   * That is, changing the center changes the transform!
   *
   * WARNING: When using the Center, we strongly recommend only changing the
   * matrix and translation to define a transform.   Changing a transform's
   * center, changes the mapping between spaces - specifically, translation is
   * not changed with respect to that new center, and so the offset is updated
   * to * maintain the consistency with translation.   If a center is not used,
   * or is set before the matrix and the offset, then it is safe to change the
   * offset directly.
   *        As a rule of thumb, if you wish to set the center explicitly, set
   * before Offset computations are done.
   *
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset.
   */
  void SetCenter( const InputPointType & center )
  {
    this->m_Center = center;
    this->ComputeOffset();
    this->Modified();
  }

  /** Get center of rotation of the AdvancedMatrixOffsetTransformBase
   *
   * This method returns the point used as the fixed
   * center of rotation for the AdvancedMatrixOffsetTransformBase.
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset.
   */
  const InputPointType & GetCenter( void ) const
  {
    return this->m_Center;
  }

  /** Set translation of an AdvancedMatrixOffsetTransformBase
   *
   * This method sets the translation of an AdvancedMatrixOffsetTransformBase.
   * This updates Offset to reflect current translation.
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset.
   */
  void SetTranslation( const OutputVectorType & translation )
  {
    this->m_Translation = translation;
    this->ComputeOffset();
    this->Modified();
  }

  /** Get translation component of the AdvancedMatrixOffsetTransformBase
   *
   * This method returns the translation used after rotation
   * about the center point.
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset.
   */
  const OutputVectorType & GetTranslation( void ) const
  {
    return this->m_Translation;
  }

  /** Set the transformation from a container of parameters.
   * The first (NOutputDimension x NInputDimension) parameters define the
   * matrix and the last NOutputDimension parameters the translation.
   * Offset is updated based on current center.
   */
  void SetParameters( const ParametersType & parameters );

  /** Get the Transformation Parameters. */
  const ParametersType & GetParameters( void ) const;

  /** Set the fixed parameters and update internal transformation. */
  virtual void SetFixedParameters( const ParametersType & );

  /** Get the Fixed Parameters. */
  virtual const ParametersType & GetFixedParameters( void ) const;

  /** Compose with another AdvancedMatrixOffsetTransformBase
   *
   * This method composes self with another AdvancedMatrixOffsetTransformBase of the
   * same dimension, modifying self to be the composition of self
   * and other.  If the argument pre is true, then other is
   * precomposed with self; that is, the resulting transformation
   * consists of first applying other to the source, followed by
   * self.  If pre is false or omitted, then other is post-composed
   * with self; that is the resulting transformation consists of
   * first applying self to the source, followed by other.
   * This updates the Translation based on current center.
   */
  void Compose( const Self * other, bool pre = 0 );

  /** Transform by an affine transformation
   *
   * This method applies the affine transform given by self to a
   * given point or vector, returning the transformed point or
   * vector.  The TransformPoint method transforms its argument as
   * an affine point, whereas the TransformVector method transforms
   * its argument as a vector.
   */
  OutputPointType     TransformPoint( const InputPointType & point ) const;
  OutputVectorType    TransformVector( const InputVectorType & vector ) const;
  OutputVnlVectorType TransformVector( const InputVnlVectorType & vector ) const;
  OutputCovariantVectorType TransformCovariantVector(
    const InputCovariantVectorType & vector ) const;

  /** Create inverse of an affine transformation
    *
    * This populates the parameters an affine transform such that
    * the transform is the inverse of self. If self is not invertible,
    * an exception is thrown.
    * Note that by default the inverese transform is centered at
    * the origin. If you need to compute the inverse centered at a point, p,
    *
    * \code
    * transform2->SetCenter( p );
    * transform1->GetInverse( transform2 );
    * \endcode
    *
    * transform2 will now contain the inverse of transform1 and will
    * with its center set to p. Flipping the two statements will produce an
    * incorrect transform.
    */
  bool GetInverse( Self * inverse ) const;

  /** \deprecated Use GetInverse instead.
   *
   * Method will eventually be made a protected member function. */
  const InverseMatrixType & GetInverseMatrix( void ) const;

  /** Indicates that this transform is linear. That is, given two
   * points P and Q, and scalar coefficients a and b, then
   *
   *           T( a*P + b*Q ) = a * T(P) + b * T(Q)
   */
  virtual bool IsLinear( void ) const
  {
    return true;
  }

  /** Indicates the category transform.
   *  e.g. an affine transform, or a local one, e.g. a deformation field.
   */
  virtual TransformCategoryType GetTransformCategory() const
  {
    return Self::Linear;
  }

  /** Compute the Jacobian of the transformation. */
  virtual void GetJacobian(
    const InputPointType &,
    JacobianType &,
    NonZeroJacobianIndicesType & ) const;

  /** Compute the spatial Jacobian of the transformation. */
  virtual void GetSpatialJacobian(
    const InputPointType &,
    SpatialJacobianType & ) const;

  /** Compute the spatial Hessian of the transformation. */
  virtual void GetSpatialHessian(
    const InputPointType &,
    SpatialHessianType & ) const;

  /** Compute the Jacobian of the spatial Jacobian of the transformation. */
  virtual void GetJacobianOfSpatialJacobian(
    const InputPointType &,
    JacobianOfSpatialJacobianType &,
    NonZeroJacobianIndicesType & ) const;

  /** Compute the Jacobian of the spatial Jacobian of the transformation. */
  virtual void GetJacobianOfSpatialJacobian(
    const InputPointType &,
    SpatialJacobianType &,
    JacobianOfSpatialJacobianType &,
    NonZeroJacobianIndicesType & ) const;

  /** Compute the Jacobian of the spatial Hessian of the transformation. */
  virtual void GetJacobianOfSpatialHessian(
    const InputPointType &,
    JacobianOfSpatialHessianType &,
    NonZeroJacobianIndicesType & ) const;

  /** Compute both the spatial Hessian and the Jacobian of the
   * spatial Hessian of the transformation. */
  virtual void GetJacobianOfSpatialHessian(
    const InputPointType & ipp,
    SpatialHessianType & sh,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

protected:
  /** Construct an AdvancedMatrixOffsetTransformBase object
   *
   * This method constructs a new AdvancedMatrixOffsetTransformBase object and
   * initializes the matrix and offset parts of the transformation
   * to values specified by the caller.  If the arguments are
   * omitted, then the AdvancedMatrixOffsetTransformBase is initialized to an identity
   * transformation in the appropriate number of dimensions.
   */
  AdvancedMatrixOffsetTransformBase( const MatrixType & matrix,
    const OutputVectorType & offset );
  AdvancedMatrixOffsetTransformBase( unsigned int paramDims );
  AdvancedMatrixOffsetTransformBase();

  /** Called by constructors: */
  virtual void PrecomputeJacobians( unsigned int paramDims );

  /** Destroy an AdvancedMatrixOffsetTransformBase object. */
  virtual ~AdvancedMatrixOffsetTransformBase() {};

  /** Print contents of an AdvancedMatrixOffsetTransformBase. */
  void PrintSelf( std::ostream &s, Indent indent ) const;

  const InverseMatrixType & GetVarInverseMatrix( void ) const
  {
    return this->m_InverseMatrix;
  };

  void SetVarInverseMatrix( const InverseMatrixType & matrix ) const
  {
    this->m_InverseMatrix = matrix;
    this->m_InverseMatrixMTime.Modified();
  };

  bool InverseMatrixIsOld( void ) const
  {
    if ( this->m_MatrixMTime != this->m_InverseMatrixMTime )
    {
      return true;
    }
    else
    {
      return false;
    }
  };

  virtual void ComputeMatrixParameters( void );

  virtual void ComputeMatrix( void );

  void SetVarMatrix( const MatrixType & matrix )
  {
    this->m_Matrix = matrix;
    this->m_MatrixMTime.Modified();
  };

  virtual void ComputeTranslation( void );

  void SetVarTranslation( const OutputVectorType & translation )
  {
    this->m_Translation = translation;
  };

  virtual void ComputeOffset( void );

  void SetVarOffset( const OutputVectorType & offset )
  {
    this->m_Offset = offset;
  };

  void SetVarCenter( const InputPointType & center )
  {
    this->m_Center = center;
  };

  /** (spatial) Jacobians and Hessians can mostly be precomputed by this transform.
   * Store them in these member variables.
   * SpatialJacobian is simply m_Matrix */
  NonZeroJacobianIndicesType m_NonZeroJacobianIndices;
  SpatialHessianType m_SpatialHessian;
  JacobianOfSpatialJacobianType m_JacobianOfSpatialJacobian;
  JacobianOfSpatialHessianType m_JacobianOfSpatialHessian;

private:

  AdvancedMatrixOffsetTransformBase(const Self & other);
  const Self & operator=( const Self & );

  /** Member variables. */
  MatrixType                  m_Matrix;         // Matrix of the transformation
  OutputVectorType            m_Offset;         // Offset of the transformation
  mutable InverseMatrixType   m_InverseMatrix;  // Inverse of the matrix
  mutable bool                m_Singular;       // Is m_Inverse singular?

  InputPointType              m_Center;
  OutputVectorType            m_Translation;

  /** To avoid recomputation of the inverse if not needed. */
  TimeStamp                   m_MatrixMTime;
  mutable TimeStamp           m_InverseMatrixMTime;

  /** Used by the GetJacobian() function which returns the
   * jacobian as an output variable. */
  mutable NonZeroJacobianIndicesType m_NonZeroJacobianIndicesTemp;

}; //class AdvancedMatrixOffsetTransformBase

}  // namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedMatrixOffsetTransformBase.txx"
#endif

#endif /* __itkAdvancedMatrixOffsetTransformBase_h */
