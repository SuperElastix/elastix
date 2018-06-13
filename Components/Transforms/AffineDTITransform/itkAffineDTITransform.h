/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkAffineDTITransform_H__
#define __itkAffineDTITransform_H__

#include "itkAffineDTI3DTransform.h"
#include "itkAdvancedMatrixOffsetTransformBase.h"

namespace itk
{

/**
 * \class AffineDTIGroup
 * \brief This class only contains a dummy class.
 *
 */

template< unsigned int Dimension >
class AffineDTIGroup
{
public:

  template< class TScalarType >
  class Dummy
  {
public:

    /** Typedef's.*/
    typedef AdvancedMatrixOffsetTransformBase< TScalarType, Dimension, Dimension > AffineDTITransform_tmp;

  };

};

/**
 * \class AffineDTIGroup<3>
 * \brief This class only contains a dummy class for the 3D case.
 *
 */

template< >
class AffineDTIGroup< 3 >
{
public:

  template< class TScalarType >
  class Dummy
  {
public:

    /** Typedef's.*/
    typedef AffineDTI3DTransform< TScalarType > AffineDTITransform_tmp;

  };

};

/**
 * \class AffineDTIGroupTemplate
 * \brief This class templates the AffineDTIGroup over its dimension.
 *
 */

template< class TScalarType, unsigned int Dimension >
class AffineDTIGroupTemplate
{
public:

  typedef AffineDTIGroupTemplate Self;
  typedef TScalarType            ScalarType;
  itkStaticConstMacro( SpaceDimension, unsigned int, Dimension );

  // This declaration of 'AffineDTI' does not work with the GCC compiler
  //    typedef AffineDTIGroup<  itkGetStaticConstMacro( SpaceDimension ) >       AffineDTI;
  // The following trick works though:
  template< unsigned int D >
  class AffineDTIGroupWrap
  {
public:

    typedef AffineDTIGroup< D > AffineDTI;
  };

  typedef AffineDTIGroupWrap< Dimension >                AffineDTIGroupWrapInstance;
  typedef typename AffineDTIGroupWrapInstance::AffineDTI AffineDTI;

  typedef typename AffineDTI::template Dummy< ScalarType > AffineDTIDummy;
  typedef typename AffineDTIDummy::AffineDTITransform_tmp  AffineDTITransform_tmp;

};

/**
 * \class AffineDTITransform
 * \brief This class makes the AffineDTI3DTransform templated over the dimension.
 *
 * This transform is an affine transform with MR-DTI specific parametrisation.
 * NB: no implementation for 2D yet!
 *
 * \ingroup Transforms
 */

template< class TScalarType, unsigned int Dimension >
class AffineDTITransform :
  public AffineDTIGroupTemplate<
  TScalarType, Dimension >::AffineDTITransform_tmp
{
public:

  /** Standard ITK-stuff. */
  typedef AffineDTITransform Self;
  typedef typename AffineDTIGroupTemplate<
    TScalarType, Dimension >
    ::AffineDTITransform_tmp Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AffineDTITransform, AffineDTIGroupTemplate );

  /** Dimension of the domain space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, Dimension );

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass::ScalarType                ScalarType;
  typedef typename Superclass::ParametersType            ParametersType;
  typedef typename Superclass::NumberOfParametersType    NumberOfParametersType;
  typedef typename Superclass::JacobianType              JacobianType;
  typedef typename Superclass::OffsetType                OffsetType;
  typedef typename Superclass::InputPointType            InputPointType;
  typedef typename Superclass::OutputPointType           OutputPointType;
  typedef typename Superclass::InputVectorType           InputVectorType;
  typedef typename Superclass::OutputVectorType          OutputVectorType;
  typedef typename Superclass::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType OutputCovariantVectorType;
  typedef typename Superclass::InputVnlVectorType        InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType       OutputVnlVectorType;

  typedef typename Superclass
    ::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType InternalMatrixType;

protected:

  AffineDTITransform(){}
  ~AffineDTITransform(){}

private:

  AffineDTITransform( const Self & ); // purposely not implemented
  void operator=( const Self & );     // purposely not implemented

};

} // end namespace itk

#endif // end #ifndef __itkAffineDTITransform_H__
