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
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkIdentityTransform.h,v $
  Language:  C++
  Date:      $Date: 2009-06-28 14:41:47 $
  Version:   $Revision: 1.19 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkAdvancedIdentityTransform_h
#define __itkAdvancedIdentityTransform_h

#include "itkObject.h"
#include "itkPoint.h"
#include "itkVector.h"
#include "itkCovariantVector.h"
#include "vnl/vnl_vector_fixed.h"
#include "itkArray.h"
#include "itkArray2D.h"
#include "itkAdvancedTransform.h"

#include "itkObjectFactory.h"

namespace itk
{

/** \class AdvancedIdentityTransform
 * \brief Implementation of an Identity Transform.
 *
 * This class defines the generic interface for an Identity Transform.
 *
 * It will map every point to itself, every vector to itself and
 * every covariant vector to itself.
 *
 * This class is intended to be used primarily as a default Transform
 * for initializing those classes supporting a generic Transform.
 *
 * This class is templated over the Representation type for coordinates
 * (that is the type used for representing the components of points and
 * vectors) and over the dimension of the space. In this case the Input
 * and Output spaces are the same so only one dimension is required.
 *
 * \ingroup Transforms
 *
 */
template< class TScalarType,
unsigned int NDimensions = 3 >
class AdvancedIdentityTransform :
  public AdvancedTransform< TScalarType, NDimensions, NDimensions >
{
public:

  /** Standard class typedefs. */
  typedef AdvancedIdentityTransform Self;
  typedef AdvancedTransform<
    TScalarType, NDimensions, NDimensions >                    Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** New method for creating an object using a factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedIdentityTransform, AdvancedTransform );

  /** Dimension of the domain space. */
  itkStaticConstMacro( InputSpaceDimension, unsigned int, NDimensions );
  itkStaticConstMacro( OutputSpaceDimension, unsigned int, NDimensions );
  itkStaticConstMacro( ParametersDimension, unsigned int, 1 );

  /** Type of the input parameters. */
  typedef  TScalarType ScalarType;

  /** Type of the input parameters. */
  typedef typename Superclass::ParametersType         ParametersType;
  typedef typename Superclass::NumberOfParametersType NumberOfParametersType;
  typedef typename Superclass::TransformCategoryType  TransformCategoryType;

  /** Type of the Jacobian matrix. */
  typedef typename Superclass::JacobianType JacobianType;

  /** Standard vector type for this class. */
  typedef Vector< TScalarType,
    itkGetStaticConstMacro( InputSpaceDimension ) >  InputVectorType;
  typedef Vector< TScalarType,
    itkGetStaticConstMacro( OutputSpaceDimension ) > OutputVectorType;

  /** Standard covariant vector type for this class */
  typedef CovariantVector< TScalarType,
    itkGetStaticConstMacro( InputSpaceDimension ) >  InputCovariantVectorType;
  typedef CovariantVector< TScalarType,
    itkGetStaticConstMacro( OutputSpaceDimension ) > OutputCovariantVectorType;

  /** Standard vnl_vector type for this class. */
  typedef vnl_vector_fixed< TScalarType,
    itkGetStaticConstMacro( InputSpaceDimension ) >  InputVnlVectorType;
  typedef vnl_vector_fixed< TScalarType,
    itkGetStaticConstMacro( OutputSpaceDimension ) > OutputVnlVectorType;

  /** Standard coordinate point type for this class */
  typedef Point< TScalarType,
    itkGetStaticConstMacro( InputSpaceDimension ) > InputPointType;
  typedef Point< TScalarType,
    itkGetStaticConstMacro( OutputSpaceDimension ) > OutputPointType;

  /** Base inverse transform type. This type should not be changed to the
   * concrete inverse transform type or inheritance would be lost.*/
  typedef typename Superclass::InverseTransformBaseType InverseTransformBaseType;
  typedef typename InverseTransformBaseType::Pointer    InverseTransformBasePointer;

  /** AdvancedTransform typedefs */
  typedef typename Superclass
    ::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType InternalMatrixType;

  /**  Method to transform a point. */
  virtual OutputPointType TransformPoint( const InputPointType  & point ) const
  { return point; }

  /**  Method to transform a vector. */
  virtual OutputVectorType TransformVector( const InputVectorType & vector ) const
  { return vector; }

  /**  Method to transform a vnl_vector. */
  virtual OutputVnlVectorType TransformVector( const InputVnlVectorType & vector ) const
  { return vector; }

  /**  Method to transform a CovariantVector. */
  virtual OutputCovariantVectorType TransformCovariantVector(
    const InputCovariantVectorType & vector ) const
  { return vector; }

  /** Set the transformation to an Identity
   *
   * This is a NULL operation in the case of this particular transform.
     The method is provided only to comply with the interface of other transforms. */
  void SetIdentity( void ) {}

  /** Return an inverse of the identity transform - another identity transform. */
  virtual InverseTransformBasePointer GetInverseTransform( void ) const
  {
    return this->New().GetPointer();
  }


  /** Indicates that this transform is linear. That is, given two
   * points P and Q, and scalar coefficients a and b, then
   *
   *           T( a*P + b*Q ) = a * T(P) + b * T(Q)
   */
  virtual bool IsLinear() const { return true; }

  /** Indicates the category transform.
   *  e.g. an affine transform, or a local one, e.g. a deformation field.
   */
  virtual TransformCategoryType GetTransformCategory() const
  {
    return Self::Linear;
  }


  /** Get the Fixed Parameters. */
  virtual const ParametersType & GetFixedParameters( void ) const
  {
    return this->m_FixedParameters;
  }


  /** Set the fixed parameters and update internal transformation. */
  virtual void SetFixedParameters( const ParametersType & ) {}

  /** Get the Parameters. */
  virtual const ParametersType & GetParameters( void ) const
  {
    return this->m_Parameters;
  }


  /** Set the fixed parameters and update internal transformation. */
  virtual void SetParameters( const ParametersType & ) {}

  /** Compute the Jacobian of the transformation. */
  virtual void GetJacobian(
    const InputPointType &,
    JacobianType & j,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
  {
    j                      = this->m_LocalJacobian;
    nonZeroJacobianIndices = this->m_NonZeroJacobianIndices;
  }


  /** Compute the spatial Jacobian of the transformation. */
  virtual void GetSpatialJacobian(
    const InputPointType &,
    SpatialJacobianType & sj ) const
  {
    sj = this->m_SpatialJacobian;
  }


  /** Compute the spatial Hessian of the transformation. */
  virtual void GetSpatialHessian(
    const InputPointType &,
    SpatialHessianType & sh ) const
  {
    sh = this->m_SpatialHessian;
  }


  /** Compute the Jacobian of the spatial Jacobian of the transformation. */
  virtual void GetJacobianOfSpatialJacobian(
    const InputPointType &,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
  {
    jsj                    = this->m_JacobianOfSpatialJacobian;
    nonZeroJacobianIndices = this->m_NonZeroJacobianIndices;
  }


  /** Compute the Jacobian of the spatial Jacobian of the transformation. */
  virtual void GetJacobianOfSpatialJacobian(
    const InputPointType &,
    SpatialJacobianType & sj,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
  {
    sj                     = this->m_SpatialJacobian;
    jsj                    = this->m_JacobianOfSpatialJacobian;
    nonZeroJacobianIndices = this->m_NonZeroJacobianIndices;
  }


  /** Compute the Jacobian of the spatial Hessian of the transformation. */
  virtual void GetJacobianOfSpatialHessian(
    const InputPointType &,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
  {
    jsh                    = this->m_JacobianOfSpatialHessian;
    nonZeroJacobianIndices = this->m_NonZeroJacobianIndices;
  }


  /** Compute both the spatial Hessian and the Jacobian of the
   * spatial Hessian of the transformation.
   */
  virtual void GetJacobianOfSpatialHessian(
    const InputPointType &,
    SpatialHessianType & sh,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
  {
    sh                     = this->m_SpatialHessian;
    jsh                    = this->m_JacobianOfSpatialHessian;
    nonZeroJacobianIndices = this->m_NonZeroJacobianIndices;
  }


protected:

  AdvancedIdentityTransform() :
    AdvancedTransform< TScalarType, NDimensions, NDimensions >( NDimensions )
  {
    // The Jacobian is constant, therefore it can be initialized in the constructor.
    this->m_LocalJacobian = JacobianType( NDimensions, 1 );
    this->m_LocalJacobian.Fill( 0.0 );

    /** SpatialJacobian is also constant. */
    this->m_SpatialJacobian.SetIdentity();

    /** Nonzero Jacobian indices, for GetJacobian. */
    this->m_NonZeroJacobianIndices.resize( ParametersDimension );
    for( unsigned int i = 0; i < ParametersDimension; ++i )
    {
      this->m_NonZeroJacobianIndices[ i ] = i;
    }

    /** Set to correct size. The elements are automatically initialized to 0. */
    this->m_HasNonZeroSpatialHessian           = false;
    this->m_HasNonZeroJacobianOfSpatialHessian = false;
    this->m_JacobianOfSpatialJacobian.resize( ParametersDimension );
    this->m_JacobianOfSpatialHessian.resize( ParametersDimension );

    /** m_SpatialHessian is automatically initialized with zeros. */
  }


  virtual ~AdvancedIdentityTransform() {}

private:

  AdvancedIdentityTransform( const Self & ); // purposely not implemented
  void operator=( const Self & );            // purposely not implemented

  JacobianType                  m_LocalJacobian;
  SpatialJacobianType           m_SpatialJacobian;
  SpatialHessianType            m_SpatialHessian;
  NonZeroJacobianIndicesType    m_NonZeroJacobianIndices;
  JacobianOfSpatialJacobianType m_JacobianOfSpatialJacobian;
  JacobianOfSpatialHessianType  m_JacobianOfSpatialHessian;

};

} // end namespace itk

#endif
