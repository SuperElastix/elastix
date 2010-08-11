/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/

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
template <class TScalarType,
  unsigned int NInputDimensions = 3,
  unsigned int NOutputDimensions = 3>
class StackTransform
  : public AdvancedTransform< TScalarType, NInputDimensions, NOutputDimensions >
{
public:
  /** Standard class typedefs. */
  typedef StackTransform              Self;
  typedef AdvancedTransform< TScalarType,
    NInputDimensions,
    NOutputDimensions >               Superclass;
  typedef SmartPointer< Self >        Pointer;
  typedef SmartPointer< const Self >  ConstPointer;

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
  typedef typename Superclass::ScalarType           ScalarType;
  typedef typename Superclass::ParametersType       ParametersType;
  typedef typename Superclass::JacobianType         JacobianType;
  typedef typename Superclass::InputPointType       InputPointType;
  typedef typename Superclass::OutputPointType      OutputPointType;
  typedef typename
    Superclass::NonZeroJacobianIndicesType					NonZeroJacobianIndicesType;

  /** Sub transform types, having a reduced dimension. */
  typedef AdvancedTransform< TScalarType,
    itkGetStaticConstMacro( ReducedInputSpaceDimension ),
    itkGetStaticConstMacro( ReducedOutputSpaceDimension ) >  SubTransformType;
  typedef typename SubTransformType::Pointer				SubTransformPointer;
  typedef std::vector< SubTransformPointer	>				SubTransformContainerType;
  typedef typename SubTransformType::JacobianType   SubTransformJacobianType;

  /** Dimension - 1 point types. */
  typedef typename SubTransformType::InputPointType     SubTransformInputPointType;
  typedef typename SubTransformType::OutputPointType    SubTransformOutputPointType;

  /**  Method to transform a point. */
  virtual OutputPointType TransformPoint( const InputPointType & ipp ) const;

  /** This returns a sparse version of the Jacobian of the transformation.
   * In this class however, the Jacobian is not sparse.
   * However, it is a useful function, since the Jacobian is passed
   * by reference, which makes it threadsafe, unlike the normal
   * GetJacobian function. */
  virtual void GetJacobian(
    const InputPointType & ipp,
    JacobianType & jac,
    NonZeroJacobianIndicesType & nzji ) const;

  /** The GetJacobian from the superclass. */
  virtual const JacobianType & GetJacobian( const InputPointType & ipp) const;

  /** Set the parameters. Checks if the number of parameters
   * is correct and sets parameters of sub transforms. */
  virtual void SetParameters( const ParametersType & param );

  /** Get the parameters. Concatenates the parameters of the
   * sub transforms. */
  virtual const ParametersType & GetParameters ( void ) const;

  /** Return the number of sub transforms that have been set. */
  virtual unsigned int GetNumberOfParameters(void) const
  {	
    if ( this->m_SubTransformContainer.size() == 0 )
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
    if ( this->m_NumberOfSubTransforms != num )
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
  virtual void SetAllSubTransforms( SubTransformType * transform ) {
    for ( unsigned int t = 0; t < this->m_NumberOfSubTransforms; ++t )
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
  virtual unsigned long GetNumberOfNonZeroJacobianIndices( void ) const;

protected:
  StackTransform();
  virtual ~StackTransform() {};

private:

  StackTransform(const Self&);  // purposely not implemented
  void operator=(const Self&);  // purposely not implemented

  // Number of transforms and transform container
  unsigned int m_NumberOfSubTransforms;
  SubTransformContainerType	 m_SubTransformContainer;

  // Stack spacing and origin of last dimension
  TScalarType m_StackSpacing, m_StackOrigin;

}; // end class StackTransform

} // end namespace itk

// \todo: copied the below from itk. does this just work like this?:

// Define instantiation macro for this template.
#define ITK_TEMPLATE_StackTransform(_, EXPORT, x, y) namespace itk { \
  _(3(class EXPORT StackTransform< ITK_TEMPLATE_3 x >)) \
  namespace Templates { typedef StackTransform< ITK_TEMPLATE_3 x > StackTransform##y; } \
  }

#if ITK_TEMPLATE_EXPLICIT
# include "Templates/itkStackTransform+-.h"
#endif

#if ITK_TEMPLATE_TXX
# include "itkStackTransform.txx"
#endif

#endif
