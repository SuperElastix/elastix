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
#ifndef __itkStackTransform_h
#define __itkStackTransform_h

#include "itkAdvancedTransform.h"
#include "itkIndex.h"

namespace itk
{

/** \class StackTransform
 * \brief Implements stack of transforms: one for every last dimension index.
 *
 * A list of transforms with dimension of Dimension - 1 is maintained:
 * one for every last dimension index. This transform selects the right
 * transform based on the last dimension index of the input point.
 *
 * \ingroup Transforms
 *
 */
template< class TScalarType,
unsigned int NInputDimensions  = 3,
unsigned int NOutputDimensions = 3 >
class StackTransform :
  public AdvancedTransform< TScalarType, NInputDimensions, NOutputDimensions >
{
public:

  /** Standard class typedefs. */
  typedef StackTransform Self;
  typedef AdvancedTransform< TScalarType,
    NInputDimensions,
    NOutputDimensions >               Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** New method for creating an object using a factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( StackTransform, AdvancedTransform );

  /** (Reduced) dimension of the domain space. */
  itkStaticConstMacro( InputSpaceDimension, unsigned int, NInputDimensions );
  itkStaticConstMacro( OutputSpaceDimension, unsigned int, NOutputDimensions );
  itkStaticConstMacro( ReducedInputSpaceDimension, unsigned int, NInputDimensions - 1 );
  itkStaticConstMacro( ReducedOutputSpaceDimension, unsigned int, NOutputDimensions - 1 );

  /** Typedefs from the Superclass. */
  typedef typename Superclass::ScalarType     ScalarType;
  typedef typename Superclass::ParametersType ParametersType;
  typedef typename Superclass::NumberOfParametersType
    NumberOfParametersType;
  typedef typename Superclass::ParametersValueType ParametersValueType;
  typedef typename Superclass::JacobianType        JacobianType;
  typedef typename Superclass::SpatialJacobianType SpatialJacobianType;
  typedef typename Superclass::JacobianOfSpatialJacobianType
    JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType SpatialHessianType;
  typedef typename Superclass::JacobianOfSpatialHessianType
    JacobianOfSpatialHessianType;
  typedef typename
    Superclass::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
  typedef typename Superclass::InputPointType      InputPointType;
  typedef typename Superclass::InputVectorType     InputVectorType;
  typedef typename Superclass::OutputVectorType    OutputVectorType;
  typedef typename Superclass::InputVnlVectorType  InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType OutputVnlVectorType;
  typedef typename Superclass::OutputCovariantVectorType
    OutputCovariantVectorType;
  typedef typename Superclass::InputCovariantVectorType
    InputCovariantVectorType;
  typedef typename Superclass::OutputPointType       OutputPointType;
  typedef typename Superclass::OutputVectorPixelType OutputVectorPixelType;
  typedef typename Superclass::InputVectorPixelType  InputVectorPixelType;

  /** Sub transform types, having a reduced dimension. */
  typedef AdvancedTransform< TScalarType,
    itkGetStaticConstMacro( ReducedInputSpaceDimension ),
    itkGetStaticConstMacro( ReducedOutputSpaceDimension ) >  SubTransformType;
  typedef typename SubTransformType::Pointer             SubTransformPointer;
  typedef std::vector< SubTransformPointer  >            SubTransformContainerType;
  typedef typename SubTransformType::JacobianType        SubTransformJacobianType;
  typedef typename SubTransformType::SpatialJacobianType SubTransformSpatialJacobianType;
  typedef typename SubTransformType::JacobianOfSpatialJacobianType
    SubTransformTypeJacobianOfSpatialJacobianType;
  typedef typename SubTransformType::SpatialHessianType           SubTransformSpatialHessianType;
  typedef typename SubTransformType::JacobianOfSpatialHessianType SubTransformJacobianOfSpatialHessianType;

  /** Dimension - 1 point types. */
  typedef typename SubTransformType::InputPointType  SubTransformInputPointType;
  typedef typename SubTransformType::OutputPointType SubTransformOutputPointType;

  /** Array type for parameter vector instantiation. */
  typedef typename ParametersType::ArrayType ParametersArrayType;

  /**  Method to transform a point. */
  virtual OutputPointType TransformPoint( const InputPointType & ipp ) const;

  /** These vector transforms are not implemented for this transform. */
  virtual OutputVectorType TransformVector( const InputVectorType & ) const
  {
    itkExceptionMacro(
        << "TransformVector(const InputVectorType &) is not implemented "
        << "for StackTransform" );
  }


  virtual OutputVnlVectorType TransformVector( const InputVnlVectorType & ) const
  {
    itkExceptionMacro(
        << "TransformVector(const InputVnlVectorType &) is not implemented "
        << "for StackTransform" );
  }


  virtual OutputCovariantVectorType TransformCovariantVector( const InputCovariantVectorType & ) const
  {
    itkExceptionMacro(
        << "TransformCovariantVector(const InputCovariantVectorType &) is not implemented "
        << "for StackTransform" );
  }


  /** This returns a sparse version of the Jacobian of the transformation.
   * In this class however, the Jacobian is not sparse.
   * However, it is a useful function, since the Jacobian is passed
   * by reference, which makes it threadsafe, unlike the normal
   * GetJacobian function. */
  virtual void GetJacobian(
    const InputPointType & ipp,
    JacobianType & jac,
    NonZeroJacobianIndicesType & nzji ) const;

  /** Compute the Spatial Jacobian. */
  virtual void GetSpatialJacobian(
    const InputPointType & ipp,
    SpatialJacobianType & sj ) const;

  /** Compute the Jacobian of the spatial Jacobian. */
  virtual void GetJacobianOfSpatialJacobian(
    const InputPointType & ipp,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Compute the spatial Jacobian and its Jacobian. */
  virtual void GetJacobianOfSpatialJacobian(
    const InputPointType & ipp, SpatialJacobianType & sj,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Compute the spatial Hessian of the transformation. */
  virtual void GetSpatialHessian(
    const InputPointType & ipp,
    SpatialHessianType & sh ) const;

  /** Compute the jacobian of the spatial Hessian of the transformation. */
  void GetJacobianOfSpatialHessian(
    const InputPointType & ipp,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nzji ) const;

  /** Compute the spatial Hessian (and its jacobian) of the transformation. */
  void GetJacobianOfSpatialHessian(
    const InputPointType & ipp,
    SpatialHessianType & sh,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nzji ) const;

  /** Set the parameters. Checks if the number of parameters
   * is correct and sets parameters of sub transforms. */
  virtual void SetParameters( const ParametersType & param );

  /** Get the parameters. Concatenates the parameters of the
   * sub transforms. */
  virtual const ParametersType & GetParameters( void ) const;

  /** Set the fixed parameters. */
  virtual void SetFixedParameters( const ParametersType & )
  {
    // \todo: to be implemented by Coert
  }


  /** Get the Fixed Parameters. */
  virtual const ParametersType & GetFixedParameters( void ) const
  {
    // \todo: to be implemented by Coert: check this:
    return this->m_FixedParameters;
  }


  /** Return the number of sub transforms that have been set. */
  virtual NumberOfParametersType GetNumberOfParameters( void ) const
  {
    if( this->m_SubTransformContainer.size() == 0 )
    {
      return 0;
    }
    else
    {
      return this->m_SubTransformContainer.size() * m_SubTransformContainer[ 0 ]->GetNumberOfParameters();
    }
  }


  /** Set/get number of transforms needed. */
  virtual void SetNumberOfSubTransforms( const unsigned int num )
  {
    if( this->m_NumberOfSubTransforms != num )
    {
      this->m_NumberOfSubTransforms = num;
      this->m_SubTransformContainer.clear();
      this->m_SubTransformContainer.resize( num );
      this->Modified();
    }
  }


  itkGetMacro( NumberOfSubTransforms, unsigned int );

  /** Set/get stack transform parameters. */
  itkSetMacro( StackSpacing, TScalarType );
  itkGetConstMacro( StackSpacing, TScalarType );
  itkSetMacro( StackOrigin, TScalarType );
  itkGetConstMacro( StackOrigin, TScalarType );

  /** Set the initial transform for sub transform i. */
  virtual void SetSubTransform( unsigned int i, SubTransformType * transform )
  {
    this->m_SubTransformContainer[ i ] = transform;
    this->Modified();
  }


  /** Set all sub transforms to transform. */
  virtual void SetAllSubTransforms( SubTransformType * transform )
  {
    for( unsigned int t = 0; t < this->m_NumberOfSubTransforms; ++t )
    {
      // Copy transform
      SubTransformPointer transformcopy = dynamic_cast< SubTransformType * >( transform->CreateAnother().GetPointer() );
      transformcopy->SetFixedParameters( transform->GetFixedParameters() );
      transformcopy->SetParameters( transform->GetParameters() );
      // Set sub transform
      this->m_SubTransformContainer[ t ] = transformcopy;
    }
  }


  /** Get a sub transform. */
  virtual SubTransformPointer GetSubTransform( unsigned int i )
  {
    return this->m_SubTransformContainer[ i ];
  }


  /** Get number of nonzero Jacobian indices. */
  virtual NumberOfParametersType GetNumberOfNonZeroJacobianIndices( void ) const;

protected:

  StackTransform();
  virtual ~StackTransform() {}

private:

  StackTransform( const Self & );  // purposely not implemented
  void operator=( const Self & );  // purposely not implemented

  // Number of transforms and transform container
  unsigned int              m_NumberOfSubTransforms;
  SubTransformContainerType m_SubTransformContainer;

  // Stack spacing and origin of last dimension
  TScalarType m_StackSpacing, m_StackOrigin;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkStackTransform.hxx"
#endif

#endif
