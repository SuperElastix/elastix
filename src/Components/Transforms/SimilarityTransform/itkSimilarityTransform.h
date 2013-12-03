/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkSimilarityTransform_H__
#define __itkSimilarityTransform_H__

#include "itkAdvancedSimilarity2DTransform.h"
#include "itkAdvancedSimilarity3DTransform.h"
#include "itkAdvancedMatrixOffsetTransformBase.h"

namespace itk
{

  /**
   * \class SimilarityGroup
   * \brief This class only contains a dummy class.
   *
   */

  template< unsigned int Dimension >
    class SimilarityGroup
  {
  public:

    template< class TScalarType >
      class Dummy
    {
    public:

      /** Typedef's.*/
      typedef AdvancedMatrixOffsetTransformBase<
        TScalarType, Dimension, Dimension >            SimilarityTransform_tmp;

    };

  };


  /**
   * \class SimilarityGroup<2>
   * \brief This class only contains a dummy class for the 2D case.
   *
   */

  template<>
    class SimilarityGroup<2>
  {
  public:

    template< class TScalarType >
      class Dummy
    {
    public:

      /** Typedef's.*/
      typedef AdvancedSimilarity2DTransform< TScalarType >      SimilarityTransform_tmp;

    };

  };


  /**
   * \class SimilarityGroup<3>
   * \brief This class only contains a dummy class for the 3D case.
   *
   */

  template<>
    class SimilarityGroup<3>
  {
  public:

    template< class TScalarType >
      class Dummy
    {
    public:

      /** Typedef's.*/
      typedef AdvancedSimilarity3DTransform< TScalarType >                  SimilarityTransform_tmp;

    };

  };


  /**
   * \class SimilarityGroupTemplate
   * \brief This class templates the SimilarityGroup over its dimension.
   *
   */

  template< class TScalarType, unsigned int Dimension >
    class SimilarityGroupTemplate
  {
  public:

    typedef SimilarityGroupTemplate Self;
    typedef TScalarType ScalarType;
    itkStaticConstMacro( SpaceDimension, unsigned int, Dimension );

    // This declaration of 'Similarity' does not work with the GCC compiler
    //    typedef SimilarityGroup<  itkGetStaticConstMacro( SpaceDimension ) >        Similarity;
    // The following trick works though:
    template <unsigned int D>
      class SimilarityGroupWrap
    {
    public:
      typedef SimilarityGroup<D> Similarity;
    };
    typedef SimilarityGroupWrap<Dimension>                      SimilarityGroupWrapInstance;
    typedef typename SimilarityGroupWrapInstance::Similarity    Similarity;

    typedef typename Similarity::template Dummy< ScalarType >   SimilarityDummy;
    typedef typename SimilarityDummy::SimilarityTransform_tmp   SimilarityTransform_tmp;

  };


  /**
   * \class SimilarityTransform
   * \brief This class combines the Similarity2DTransform with the Similarity3DTransform.
   *
   * This transform is a rigid body transformation, with a uniform scaling.
   *
   * \ingroup Transforms
   */

  template< class TScalarType, unsigned int Dimension >
    class SimilarityTransform:
  public SimilarityGroupTemplate<
    TScalarType, Dimension >::SimilarityTransform_tmp
  {
  public:

    /** Standard ITK-stuff. */
    typedef SimilarityTransform                 Self;
    typedef typename SimilarityGroupTemplate<
      TScalarType, Dimension >
      ::SimilarityTransform_tmp                 Superclass;
    typedef SmartPointer<Self>                  Pointer;
    typedef SmartPointer<const Self>            ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( SimilarityTransform, SimilarityGroupTemplate );

    /** Dimension of the domain space. */
    itkStaticConstMacro( SpaceDimension, unsigned int, Dimension );

    /** Typedefs inherited from the superclass. */

    /** These are both in Similarity2D and Similarity3D. */
    typedef typename Superclass::ScalarType                   ScalarType;
    typedef typename Superclass::ParametersType               ParametersType;
    typedef typename Superclass::NumberOfParametersType       NumberOfParametersType;
    typedef typename Superclass::JacobianType                 JacobianType;
    typedef typename Superclass::OffsetType                   OffsetType;
    typedef typename Superclass::InputPointType               InputPointType;
    typedef typename Superclass::OutputPointType              OutputPointType;
    typedef typename Superclass::InputVectorType              InputVectorType;
    typedef typename Superclass::OutputVectorType             OutputVectorType;
    typedef typename Superclass::InputCovariantVectorType     InputCovariantVectorType;
    typedef typename Superclass::OutputCovariantVectorType    OutputCovariantVectorType;
    typedef typename Superclass::InputVnlVectorType           InputVnlVectorType;
    typedef typename Superclass::OutputVnlVectorType          OutputVnlVectorType;

    typedef typename Superclass
      ::NonZeroJacobianIndicesType                    NonZeroJacobianIndicesType;
    typedef typename Superclass::SpatialJacobianType  SpatialJacobianType;
    typedef typename Superclass
      ::JacobianOfSpatialJacobianType                 JacobianOfSpatialJacobianType;
    typedef typename Superclass::SpatialHessianType   SpatialHessianType;
    typedef typename Superclass
      ::JacobianOfSpatialHessianType                  JacobianOfSpatialHessianType;
    typedef typename Superclass::InternalMatrixType   InternalMatrixType;

  protected:

    SimilarityTransform(){};
    ~SimilarityTransform(){};

  private:

    SimilarityTransform( const Self& ); // purposely not implemented
    void operator=( const Self& ); // purposely not implemented

  };


} // end namespace itk

#endif // end #ifndef __itkSimilarityTransform_H__

