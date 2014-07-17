/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkPolydataDummyPenalty_h
#define __itkPolydataDummyPenalty_h

#include "itkSingleValuedPointSetToPointSetMetric.h"
#include "itkPoint.h"
#include "itkPointSet.h"
#include "itkImage.h"
#include "itkMesh.h"
#include <itkVectorContainer.h>


//#include <vnl/vnl_math.h>
//#include <vnl/vnl_matrix.h>

namespace itk
{

  /** \class MeshPenalty
  * \brief A dummy metric to generate transformed meshes each iteration.
  *
  *
  *
  * \ingroup RegistrationMetrics
  */
  template < class TFixedPointSet, class TMovingPointSet >
  class ITK_EXPORT MeshPenalty
    : public SingleValuedPointSetToPointSetMetric<TFixedPointSet, TMovingPointSet>
  {
  public:


    /** Standard class typedefs. */
    typedef MeshPenalty                  Self;
    typedef SingleValuedPointSetToPointSetMetric<
      TFixedPointSet, TMovingPointSet >               Superclass;
    typedef SmartPointer<Self>                          Pointer;
    typedef SmartPointer<const Self>                    ConstPointer;

    /** Type used for representing point components  */

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( MeshPenalty,
      SingleValuedPointSetToPointSetMetric );

    /** Types transferred from the base class */
    typedef typename Superclass::TransformType              TransformType;
    typedef typename Superclass::TransformPointer           TransformPointer;
    typedef typename Superclass::TransformParametersType    TransformParametersType;
    typedef typename Superclass::TransformJacobianType      TransformJacobianType;

    typedef typename Superclass::MeasureType                MeasureType;
    typedef typename Superclass::DerivativeType             DerivativeType;
    typedef typename Superclass::DerivativeValueType        DerivativeValueType;


    typedef typename Superclass::InputPointType             InputPointType;

    typedef typename OutputPointType::CoordRepType          CoordRepType;
    typedef vnl_vector<CoordRepType>                        VnlVectorType;

    typedef typename TransformType::InputPointType                FixedImagePointType;
    typedef typename TransformType::OutputPointType               MovingImagePointType;


    /** Typedefs. */  
    typedef typename TransformType::SpatialJacobianType      SpatialJacobianType;

    typedef itk::Vector<typename TransformType::ScalarType, FixedPointSetDimension>                  PointNormalType;

    typedef unsigned char                                               DummyMeshPixelType;
    typedef DefaultStaticMeshTraits<PointNormalType, 
      FixedPointSetDimension, FixedPointSetDimension, CoordRepType>     MeshTraitsType;
    typedef typename Mesh<PointNormalType, FixedPointSetDimension,
      MeshTraitsType >                                                  FixedMeshType;


    typedef typename FixedMeshType::ConstPointer                        FixedMeshConstPointer;
    typedef typename FixedMeshType::Pointer                             FixedMeshPointer;
    typedef typename MeshTraitsType::CellType::CellInterface            CellInterfaceType;

    typedef typename FixedMeshType::PointType                           MeshPointType;
    typedef typename FixedMeshType::PointType::VectorType               VectorType;

    typedef typename FixedMeshType::PointsContainer                     MeshPointsContainerType;
    typedef typename MeshPointsContainerType::Pointer                   MeshPointsContainerPointer;
    typedef typename MeshPointsContainerType::ConstPointer              MeshPointsContainerConstPointer;
    typedef typename FixedMeshType::PointsContainerConstIterator        MeshPointsContainerConstIteratorType;
    typedef typename FixedMeshType::PointsContainerIterator             MeshPointsContainerIteratorType;
    
    
    typedef typename FixedMeshType::PointDataContainer                  MeshPointDataContainerType;
    typedef typename FixedMeshType::PointDataContainerConstPointer      MeshPointDataContainerConstPointer;
    typedef typename FixedMeshType::PointDataContainerPointer           MeshPointDataContainerPointer;
    //typedef typename FixedMeshType::PointDataContainerConstIterator     MeshPointDataContainerConstIteratorType;
    typedef typename FixedMeshType::PointDataContainerIterator          MeshPointDataContainerConstIteratorType;  
    typedef typename MeshPointDataContainerType::Iterator               MeshPointDataContainerIteratorType;
    

    typedef unsigned int                                                MeshIdType;
    typedef typename VectorContainer<MeshIdType, FixedMeshConstPointer>     FixedMeshContainerType;
    typedef typename FixedMeshContainerType::Pointer                    FixedMeshContainerPointer;
    typedef typename FixedMeshContainerType::ConstPointer               FixedMeshContainerConstPointer;


    typedef typename VectorContainer<MeshIdType, FixedMeshPointer>       MappedMeshContainerType;
    typedef typename MappedMeshContainerType::Pointer                    MappedMeshContainerPointer;
    typedef typename MappedMeshContainerType::ConstPointer               MappedMeshContainerConstPointer;    



    typedef typename Array< DerivativeValueType >                                          MeshPointsDerivativeValueType;

    itkSetConstObjectMacro( FixedMeshContainer, FixedMeshContainerType);
    itkGetConstObjectMacro( FixedMeshContainer, FixedMeshContainerType );


    itkSetObjectMacro( MappedMeshContainer, MappedMeshContainerType );
    itkGetObjectMacro( MappedMeshContainer, MappedMeshContainerType );

    /** Get the mapped points. */
    //itkGetObjectMacro( MappedPoints, MeshPointsContainerPointer );

    /** Connect the fixed pointset.  */
    //itkSetConstObjectMacro( FixedMesh, FixedMeshType );

    /** Get the fixed pointset. */
    //itkGetConstObjectMacro( FixedMesh, FixedMeshType );

    /** Connect the Transform. */
    //itkSetObjectMacro( Transform, TransformType );

    /** Get a pointer to the Transform.  */
    //itkGetConstObjectMacro( Transform, TransformType );

    /** Set the parameters defining the Transform. */
    //void SetTransformParameters( const ParametersType & parameters ) const;

    /** Return the number of parameters required by the transform. */
    //unsigned int GetNumberOfParameters( void ) const
    //{ return this->m_Transform->GetNumberOfParameters(); }

    /** Initialize the Metric by making sure that all the components are
    *  present and plugged together correctly.
    */
    virtual void Initialize( void ) throw ( ExceptionObject );

    /** Set the fixed mask. */
    // \todo: currently not used
    //itkSetConstObjectMacro( FixedImageMask, FixedImageMaskType );

    /** Get the fixed mask. */
    //itkGetConstObjectMacro( FixedImageMask, FixedImageMaskType );


    /**  Get the value for single valued optimizers. */
    MeasureType GetValue( const TransformParametersType & parameters ) const;

    /** Get the derivatives of the match measure. */
    void GetDerivative( const TransformParametersType & parameters,
      DerivativeType & Derivative ) const;

    /**  Get value and derivatives for multiple valued optimizers. */
    void GetValueAndDerivative( const TransformParametersType & parameters,
      MeasureType& Value, DerivativeType& Derivative ) const;

  protected:
    MeshPenalty();
    virtual ~MeshPenalty();


    /** PrintSelf. */
    void PrintSelf(std::ostream& os, Indent indent) const;

    /** Member variables. */
    mutable FixedMeshContainerConstPointer  m_FixedMeshContainer;
    mutable MappedMeshContainerPointer  m_MappedMeshContainer;

    CoordRepType m_KernelWidth;
    CoordRepType m_CutOffDist;
  private:

    MeshPenalty(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented


  }; // end class MeshPenalty

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkPolydataDummyPenalty.hxx"
#endif

#endif
