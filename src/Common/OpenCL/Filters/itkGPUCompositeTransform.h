/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUCompositeTransform_h
#define __itkGPUCompositeTransform_h

#include "itkCompositeTransform.h"
#include "itkGPUTransformBase.h"

namespace itk
{
/** \class GPUCompositeTransform
 */
template< class TScalarType = float, unsigned int NDimensions = 3,
          class TParentImageFilter = CompositeTransform< TScalarType, NDimensions > >
class GPUCompositeTransform : public TParentImageFilter, public GPUTransformBase
{
public:
  /** Standard class typedefs. */
  typedef GPUCompositeTransform      Self;
  typedef TParentImageFilter         Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUCompositeTransform, TParentImageFilter );

  /** Type of the scalar representing coordinate and vector elements. */
  typedef typename Superclass::ScalarType    ScalarType;
  typedef typename Superclass::TransformType TransformType;

  /** Dimension of the domain space. */
  itkStaticConstMacro( InputSpaceDimension, unsigned int, NDimensions );
  itkStaticConstMacro( OutputSpaceDimension, unsigned int, NDimensions );

  /**  */
  bool HasIdentityTransform() const;

  /**  */
  bool HasMatrixOffsetTransform() const;

  /**  */
  bool HasTranslationTransform() const;

  /**  */
  bool HasBSplineTransform() const;

  /**  */
  bool IsIdentityTransform( const size_t index ) const;

  /**  */
  bool IsMatrixOffsetTransform( const size_t index ) const;

  /**  */
  bool IsTranslationTransform( const size_t index ) const;

  /**  */
  bool IsBSplineTransform( const size_t index ) const;

protected:
  GPUCompositeTransform();
  virtual ~GPUCompositeTransform() {}
  void PrintSelf( std::ostream & s, Indent indent ) const;

  virtual bool GetSourceCode( std::string & _source ) const;

  virtual GPUDataManager::Pointer GetParametersDataManager( const size_t index ) const;

private:
  GPUCompositeTransform( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );      // purposely not implemented

  bool IsIdentityTransform( const size_t index,
                            const bool _loadSource, std::string & _source ) const;

  bool IsMatrixOffsetTransform( const size_t index,
                                const bool _loadSource, std::string & _source ) const;

  bool IsTranslationTransform( const size_t index,
                               const bool _loadSource, std::string & _source ) const;

  bool IsBSplineTransform( const size_t index,
                           const bool _loadSource, std::string & _source ) const;

  std::vector< std::string > m_Sources;
  bool                       m_SourcesLoaded;
};

/** \class GPUCompositeTransformFactory
* \brief Object Factory implementation for GPUCompositeTransform
*/
class GPUCompositeTransformFactory : public ObjectFactoryBase
{
public:
  typedef GPUCompositeTransformFactory Self;
  typedef ObjectFactoryBase            Superclass;
  typedef SmartPointer< Self >         Pointer;
  typedef SmartPointer< const Self >   ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char * GetDescription() const { return "A Factory for GPUCompositeTransform"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUCompositeTransformFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPUCompositeTransformFactory::Pointer factory =
      GPUCompositeTransformFactory::New();
    ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPUCompositeTransformFactory( const Self & ); // purposely not implemented
  void operator=( const Self & );               // purposely not implemented

#define OverrideCompositeTransformTypeMacro( st, dm )                   \
  {                                                                     \
    this->RegisterOverride(                                             \
      typeid( CompositeTransform< st, dm > ).name(),                    \
      typeid( GPUCompositeTransform< st, dm > ).name(),                 \
      "GPU CompositeTransform Override",                                \
      true,                                                             \
      CreateObjectFunction< GPUCompositeTransform< st, dm > >::New() ); \
  }

  GPUCompositeTransformFactory()
  {
    if ( IsGPUAvailable() )
    {
      OverrideCompositeTransformTypeMacro( float, 1 );
      //OverrideCompositeTransformTypeMacro(double, 1);

      OverrideCompositeTransformTypeMacro( float, 2 );
      //OverrideCompositeTransformTypeMacro(double, 2);

      OverrideCompositeTransformTypeMacro( float, 3 );
      //OverrideCompositeTransformTypeMacro(double, 3);
    }
  }
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUCompositeTransform.hxx"
#endif

#endif /* __itkGPUCompositeTransform_h */
