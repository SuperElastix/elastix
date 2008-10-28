/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAdvancedAffineTransform.txx,v $
  Language:  C++
  Date:      $Date: 2006-10-14 19:58:31 $
  Version:   $Revision: 1.57 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkAdvancedAffineTransform_txx
#define __itkAdvancedAffineTransform_txx

#include "itkNumericTraits.h"
#include "itkAdvancedAffineTransform.h"
#include "vnl/algo/vnl_matrix_inverse.h"


namespace itk
{

/** Constructor with default arguments */
template<class TScalarType, unsigned int NDimensions>
AdvancedAffineTransform<TScalarType, NDimensions>::
AdvancedAffineTransform(): Superclass(SpaceDimension,ParametersDimension)
{
}


/** Constructor with default arguments */
template<class TScalarType, unsigned int NDimensions>
AdvancedAffineTransform<TScalarType, NDimensions>::
AdvancedAffineTransform( unsigned int outputSpaceDimension, 
                 unsigned int parametersDimension   ):
  Superclass(outputSpaceDimension,parametersDimension)
{
}


/** Constructor with explicit arguments */
template<class TScalarType, unsigned int NDimensions>
AdvancedAffineTransform<TScalarType, NDimensions>::
AdvancedAffineTransform(const MatrixType & matrix,
               const OutputVectorType & offset):
  Superclass(matrix, offset)
{
}


/**  Destructor */
template<class TScalarType, unsigned int NDimensions>
AdvancedAffineTransform<TScalarType, NDimensions>::
~AdvancedAffineTransform()
{
  return;
}


/** Print self */
template<class TScalarType, unsigned int NDimensions>
void
AdvancedAffineTransform<TScalarType, NDimensions>::
PrintSelf(std::ostream &os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}


/** Compose with a translation */
template<class TScalarType, unsigned int NDimensions>
void
AdvancedAffineTransform<TScalarType, NDimensions>::
Translate(const OutputVectorType &trans, bool pre)
{
  OutputVectorType newTranslation = this->GetTranslation();
  if (pre) 
    {
    newTranslation += this->GetMatrix() * trans;
    }
  else 
    {
    newTranslation += trans;
    }
  this->SetVarTranslation(newTranslation);
  this->ComputeOffset();
  this->Modified();
  return;
}


/** Compose with isotropic scaling */
template<class TScalarType, unsigned int NDimensions>
void
AdvancedAffineTransform<TScalarType, NDimensions>
::Scale(const TScalarType &factor, bool pre) 
{
  if (pre) 
    {
    MatrixType newMatrix = this->GetMatrix();
    newMatrix *= factor;
    this->SetVarMatrix(newMatrix);
    }
  else 
    {
    MatrixType newMatrix = this->GetMatrix();
    newMatrix *= factor;
    this->SetVarMatrix(newMatrix);

    OutputVectorType newTranslation = this->GetTranslation();
    newTranslation *= factor;
    this->SetVarTranslation(newTranslation);
    }
  this->ComputeMatrixParameters();
  this->ComputeOffset();
  this->Modified();
  return;
}


/** Compose with anisotropic scaling */
template<class TScalarType, unsigned int NDimensions>
void
AdvancedAffineTransform<TScalarType, NDimensions>
::Scale(const OutputVectorType &factor, bool pre) 
{
  MatrixType trans;
  unsigned int i, j;

  for (i = 0; i < NDimensions; i++) 
    {
    for (j = 0; j < NDimensions; j++) 
      {
      trans[i][j] = 0.0;
      }
    trans[i][i] = factor[i];
    }
  if (pre) 
    {
    this->SetVarMatrix( this->GetMatrix() * trans );
    }
  else 
    {
    this->SetVarMatrix( trans * this->GetMatrix() );
    this->SetVarTranslation( trans * this->GetTranslation() );
    }
  this->ComputeMatrixParameters();
  this->ComputeOffset();
  this->Modified();
  return;
}


/** Compose with elementary rotation */
template<class TScalarType, unsigned int NDimensions>
void
AdvancedAffineTransform<TScalarType, NDimensions>
::Rotate(int axis1, int axis2, TScalarType angle, bool pre) 
{
  MatrixType trans;
  unsigned int i, j;

  for (i = 0; i < NDimensions; i++) 
    {
    for (j = 0; j < NDimensions; j++) 
      {
      trans[i][j] = 0.0;
      }
    trans[i][i] = 1.0;
    }
  trans[axis1][axis1] =  vcl_cos(angle);
  trans[axis1][axis2] =  vcl_sin(angle);
  trans[axis2][axis1] = -sin(angle);
  trans[axis2][axis2] =  vcl_cos(angle);
  if (pre) 
    {
    this->SetVarMatrix( this->GetMatrix() * trans );
    }
  else 
    {
    this->SetVarMatrix( trans * this->GetMatrix() );
    this->SetVarTranslation( trans * this->GetTranslation() );
    }
  this->ComputeMatrixParameters();
  this->ComputeOffset();
  this->Modified();
  return;
}


/** Compose with 2D rotation
 * \todo Find a way to generate a compile-time error
 * is this is used with NDimensions != 2. */
template<class TScalarType, unsigned int NDimensions>
void
AdvancedAffineTransform<TScalarType, NDimensions>
::Rotate2D(TScalarType angle, bool pre)
{
  MatrixType trans;

  trans[0][0] =  vcl_cos(angle);
  trans[0][1] = -sin(angle);
  trans[1][0] = vcl_sin(angle);
  trans[1][1] =  vcl_cos(angle);
  if (pre) 
    {
    this->SetVarMatrix( this->GetMatrix() * trans );
    }
  else 
    {
    this->SetVarMatrix( trans * this->GetMatrix() );
    this->SetVarTranslation( trans * this->GetTranslation() );
    }
  this->ComputeMatrixParameters();
  this->ComputeOffset();
  this->Modified();
  return;
}


/** Compose with 3D rotation
 *  \todo Find a way to generate a compile-time error
 *  is this is used with NDimensions != 3. */
template<class TScalarType, unsigned int NDimensions>
void
AdvancedAffineTransform<TScalarType, NDimensions>
::Rotate3D(const OutputVectorType &axis, TScalarType angle, bool pre)
{
  MatrixType trans;
  ScalarType r, x1, x2, x3;
  ScalarType q0, q1, q2, q3;

  // Convert the axis to a unit vector
  r = vcl_sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
  x1 = axis[0] / r;
  x2 = axis[1] / r;
  x3 = axis[2] / r;

  // Compute quaternion elements
  q0 = vcl_cos(angle/2.0);
  q1 = x1 * vcl_sin(angle/2.0);
  q2 = x2 * vcl_sin(angle/2.0);
  q3 = x3 * vcl_sin(angle/2.0);

  // Compute elements of the rotation matrix
  trans[0][0] = q0*q0 + q1*q1 - q2*q2 - q3*q3;
  trans[0][1] = 2.0*(q1*q2 - q0*q3);
  trans[0][2] = 2.0*(q1*q3 + q0*q2);
  trans[1][0] = 2.0*(q1*q2 + q0*q3);
  trans[1][1] = q0*q0 + q2*q2 - q1*q1 - q3*q3;
  trans[1][2] = 2.0*(q2*q3 - q0*q1);
  trans[2][0] = 2.0*(q1*q3 - q0*q2);
  trans[2][1] = 2.0*(q2*q3 + q0*q1);
  trans[2][2] = q0*q0 + q3*q3 - q1*q1 - q2*q2;

  // Compose rotation matrix with the existing matrix
  if (pre) 
    {
    this->SetVarMatrix( this->GetMatrix() * trans );
    }
  else 
    {
    this->SetVarMatrix( trans * this->GetMatrix() );
    this->SetVarTranslation( trans * this->GetTranslation() );
    }
  this->ComputeMatrixParameters();
  this->ComputeOffset();
  this->Modified();
  return;
}


/** Compose with elementary rotation */
template<class TScalarType, unsigned int NDimensions>
void
AdvancedAffineTransform<TScalarType, NDimensions>
::Shear(int axis1, int axis2, TScalarType coef, bool pre)
{
  MatrixType trans;
  unsigned int i, j;

  for (i = 0; i < NDimensions; i++) 
    {
    for (j = 0; j < NDimensions; j++) 
      {
      trans[i][j] = 0.0;
      }
    trans[i][i] = 1.0;
    }
  trans[axis1][axis2] =  coef;
  if (pre) 
    {
    this->SetVarMatrix( this->GetMatrix() * trans );
    }
  else 
    {
    this->SetVarMatrix( trans * this->GetMatrix() );
    this->SetVarTranslation( trans * this->GetTranslation() );
    }
  this->ComputeMatrixParameters();
  this->ComputeOffset();
  this->Modified();
  return;
}


/** Compute a distance between two affine transforms */
template<class TScalarType, unsigned int NDimensions>
typename AdvancedAffineTransform<TScalarType, NDimensions>::ScalarType
AdvancedAffineTransform<TScalarType, NDimensions>
::Metric(const Self * other) const
{
  ScalarType result = 0.0, term;

  for (unsigned int i = 0; i < NDimensions; i++) 
    {
    for (unsigned int j = 0; j < NDimensions; j++) 
      {
      term = this->GetMatrix()[i][j] - other->GetMatrix()[i][j];
      result += term * term;
      }
    term = this->GetOffset()[i] - other->GetOffset()[i];
    result += term * term;
    }
  return vcl_sqrt(result);
}


/** Compute a distance between self and the identity transform */
template<class TScalarType, unsigned int NDimensions>
typename AdvancedAffineTransform<TScalarType, NDimensions>::ScalarType
AdvancedAffineTransform<TScalarType, NDimensions>
::Metric(void) const
{
  ScalarType result = 0.0, term;

  for (unsigned int i = 0; i < NDimensions; i++) 
    {
    for (unsigned int j = 0; j < NDimensions; j++) 
      {
      if (i == j)
        {
        term = this->GetMatrix()[i][j] - 1.0;
        }
      else
        {
        term = this->GetMatrix()[i][j];
        }
      result += term * term;
      }
    term = this->GetOffset()[i];
    result += term * term;
    }

  return vcl_sqrt(result);
}


/**
 * ********************* GetJacobian ****************************
 */

template<class TScalarType, unsigned int NDimensions>
void
AdvancedAffineTransform<TScalarType, NDimensions>
::GetJacobian(
  const InputPointType & p,
  JacobianType & j,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  j = this->Superclass::GetJacobian( p );

  unsigned int parSize = this->GetNumberOfParameters();
  nonZeroJacobianIndices.resize( parSize );
  for ( unsigned int mu = 0; mu < parSize; ++mu )
  {
    nonZeroJacobianIndices[ mu ] = mu;
  }

} // end GetJacobian()


/**
 * ********************* GetSpatialJacobian ****************************
 */

template<class TScalarType, unsigned int NDimensions>
void
AdvancedAffineTransform<TScalarType, NDimensions>
::GetSpatialJacobian(
  const InputPointType &,
  SpatialJacobianType & sj ) const
{
  /** In 2D the SpatialJacobian looks like:
   * sj = [ mu0 mu1 ]
   *      [ mu2 mu3 ]
   */

  /** Fill the matrix. *
  for ( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    for ( unsigned int j = 0; j < SpaceDimension; ++j )
    {
      sj[ i ][ j ] = this->m_Parameters[ i * SpaceDimension + j ];
    }
  }*/
  sj = this->GetMatrix(); // CHECK

} // end GetSpatialJacobian()


/**
 * ********************* GetSpatialHessian ****************************
 */

template<class TScalarType, unsigned int NDimensions>
void
AdvancedAffineTransform<TScalarType, NDimensions>
::GetSpatialHessian(
  const InputPointType &,
  SpatialHessianType & sh ) const
{
  /** The SpatialHessian contains only zeros. We simply return nothing. */
  sh.resize( 0 );

} // end GetSpatialHessian()

  
/**
 * ********************* GetJacobianOfSpatialJacobian ****************************
 */

template<class TScalarType, unsigned int NDimensions>
void
AdvancedAffineTransform<TScalarType, NDimensions>
::GetJacobianOfSpatialJacobian(
  const InputPointType &,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  unsigned int parSize = this->GetNumberOfParameters();
  jsj.resize( parSize );
  nonZeroJacobianIndices.resize( parSize );

  /** Fill the matrices. */
  for ( unsigned int mu = 0; mu < parSize; ++mu )
  {
    SpatialJacobianType sj;
    sj.Fill( 0.0 );
    if ( mu < SpaceDimension * SpaceDimension )
    {
      sj[ mu / SpaceDimension ][ mu % SpaceDimension ] = 1.0;
    }
    jsj[ mu ] = sj;
    nonZeroJacobianIndices[ mu ] = mu;
  }

} // end GetJacobianOfSpatialJacobian()


/**
 * ********************* GetJacobianOfSpatialHessian ****************************
 */

template<class TScalarType, unsigned int NDimensions>
void
AdvancedAffineTransform<TScalarType, NDimensions>
::GetJacobianOfSpatialHessian(
  const InputPointType &,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  /** The JacobianOfSpatialHessian contains only zeros.
   * We simply return nothing.
   */
  jsh.resize( 0 );
  nonZeroJacobianIndices.resize( 0 );
  
} // end GetJacobianOfSpatialHessian()


} // namespace

#endif
