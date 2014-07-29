/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxMissingStructurePenalty_H__
#define __elxMissingStructurePenalty_H__

#include "elxIncludes.h"
#include "itkMissingStructurePenalty.h"

#include "itkMeshFileReader.h"
#include "itkMeshFileWriter.h"

namespace elastix
{

/**
 * \class MissingStructurePenalty
 * \brief .
 *
 * \brief Computes the (pseudo) volume of the transformed surface mesh of a structure.\n
 * A metric based on the itk::MissingStructurePenalty.\n
 * \author F.F. Berendsen, Image Sciences Institute, UMC Utrecht, The Netherlands
 * \note If you use the MissingStructurePenalty anywhere we would appreciate if you cite the following article:\n
 * F.F. Berendsen, A.N.T.J. Kotte, A.A.C. de Leeuw, I.M. Jürgenliemk-Schulz,\n
 * M.A. Viergever and J.P.W. Pluim "Registration of structurally dissimilar \n
 * images in MRI-based brachytherapy ", Phys. Med. Biol. 59 (2014) 4033-4045.\n
 * http://stacks.iop.org/0031-9155/59/4033
 * The parameters used in this class are:
 * \parameter Metric: Select this metric as follows:\n
 *    <tt>(Metric "MissingStructurePenalty")</tt>
 * \parameter
 *    <tt>(WriteResultMeshAfterEachIteration "True")</tt>
 * \parameter
 *    <tt>(WriteResultMeshAfterEachResolution "True")</tt>
 * The command-line options for input meshes is: -fmesh<[A-Z]><MetricNumber>.
 * \ingroup RegistrationMetrics
 */

template< class TElastix >
class MissingStructurePenalty :
  public
  itk::MissingVolumeMeshPenalty<
  typename MetricBase< TElastix >::FixedPointSetType,
  typename MetricBase< TElastix >::MovingPointSetType >,
  public MetricBase< TElastix >
{
public:

  /** Standard ITK-stuff. */
  typedef MissingStructurePenalty Self;
  typedef itk::MissingVolumeMeshPenalty<
    typename MetricBase< TElastix >::FixedPointSetType,
    typename MetricBase< TElastix >::MovingPointSetType > Superclass1;
  typedef MetricBase< TElastix >          Superclass2;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( MissingStructurePenalty,
    itk::MissingVolumeMeshPenalty );

  /** Name of this class.
  * Use this name in the parameter file to select this specific metric. \n
  * example: <tt>(Metric "MissingStructurePenalty")</tt>\n
  */
  elxClassNameMacro( "MissingStructurePenalty" );

  /** Typedefs from the superclass. */
  typedef typename Superclass1::FixedMeshType         FixedMeshType;
  typedef typename Superclass1::FixedMeshPointer      FixedMeshPointer;
  typedef typename Superclass1::FixedMeshConstPointer FixedMeshConstPointer;

  typedef typename Superclass1::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass1::FixedPointSetType            FixedPointSetType;
  typedef typename Superclass1::FixedPointSetConstPointer    FixedPointSetConstPointer;
  typedef typename Superclass1::FixedMeshContainerType       FixedMeshContainerType;
  typedef typename Superclass1::FixedMeshContainerPointer    FixedMeshContainerPointer;
  typedef typename Superclass1::MappedMeshContainerType      MappedMeshContainerType;
  typedef typename Superclass1::MappedMeshContainerPointer   MappedMeshContainerPointer;
  typedef typename Superclass1::MovingPointSetType           MovingPointSetType;
  typedef typename Superclass1::MovingPointSetConstPointer   MovingPointSetConstPointer;
  typedef typename Superclass1::CellInterfaceType            CellInterfaceType;

  //  typedef typename Superclass1::FixedImageRegionType       FixedImageRegionType;
  typedef typename Superclass1::TransformType           TransformType;
  typedef typename Superclass1::TransformPointer        TransformPointer;
  typedef typename Superclass1::InputPointType          InputPointType;
  typedef typename Superclass1::OutputPointType         OutputPointType;
  typedef typename Superclass1::TransformParametersType TransformParametersType;
  typedef typename Superclass1::TransformJacobianType   TransformJacobianType;
  //  typedef typename Superclass1::RealType                   RealType;
  typedef typename Superclass1::FixedImageMaskType     FixedImageMaskType;
  typedef typename Superclass1::FixedImageMaskPointer  FixedImageMaskPointer;
  typedef typename Superclass1::MovingImageMaskType    MovingImageMaskType;
  typedef typename Superclass1::MovingImageMaskPointer MovingImageMaskPointer;
  typedef typename Superclass1::MeasureType            MeasureType;
  typedef typename Superclass1::DerivativeType         DerivativeType;
  typedef typename Superclass1::ParametersType         ParametersType;

  typedef typename OutputPointType::CoordRepType CoordRepType;

  typedef typename Superclass1::MeshIdType MeshIdType;
  /** Other typedef's. */
  typedef itk::Object ObjectType;

  typedef itk::AdvancedCombinationTransform< CoordRepType,
    itkGetStaticConstMacro( FixedImageDimension ) >   CombinationTransformType;
  typedef typename
    CombinationTransformType::InitialTransformType InitialTransformType;

  /** Typedefs inherited from elastix. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;
  typedef typename Superclass2::FixedImageType       FixedImageType;
  typedef typename Superclass2::MovingImageType      MovingImageType;

  /** The fixed image dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    FixedImageType::ImageDimension );
  itkStaticConstMacro( MovingImageDimension, unsigned int,
    MovingImageType::ImageDimension );

  /** Assuming fixed and moving pointsets are of equal type, which implicitly
  * assumes that the fixed and moving image are of the same type.
  */
  typedef FixedPointSetType       PointSetType;
  typedef FixedMeshType  MeshType;
  typedef FixedImageType ImageType;

  /** Sets up a timer to measure the initialization time and calls the
  * Superclass' implementation.
  */
  virtual void Initialize( void ) throw ( itk::ExceptionObject );

  /**
  * Do some things before registration:
  * \li Load and set the pointsets.
  */
  virtual int BeforeAllBase( void );

  virtual void BeforeRegistration( void );

  virtual void BeforeEachResolution( void );

  virtual void AfterEachIteration( void );

  virtual void AfterEachResolution( void );

  /** Function to read the corresponding points. */
  unsigned int ReadMesh(
  const std::string & meshFileName,
  typename FixedMeshType::Pointer & mesh );

  void WriteResultMesh( const char * filename, MeshIdType meshId );

  unsigned int ReadTransformixPoints( const std::string & filename, typename MeshType::Pointer & mesh );

  /** Overwrite to silence warning. */
  virtual void SelectNewSamples( void ){}

protected:

  /** The constructor. */
  MissingStructurePenalty(){}
  /** The destructor. */
  virtual ~MissingStructurePenalty() {}

private:

  /** The private constructor. */
  MissingStructurePenalty( const Self & );  // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );           // purposely not implemented

  unsigned int m_NumberOfMeshes;
}; // end class MissingStructurePenalty

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxMissingStructurePenalty.hxx"
#endif

#endif // end #ifndef __elxMissingStructurePenalty_H__

