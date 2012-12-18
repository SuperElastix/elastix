/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUTranslationTransform_h
#define __itkGPUTranslationTransform_h

#include "itkTranslationTransform.h"
#include "itkGPUTranslationTransformBase.h"

namespace itk
{
/** \class GPUTranslationTransform
* \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
* Department of Radiology, Leiden, The Netherlands
*
* This implementation was taken from elastix (http://elastix.isi.uu.nl/).
*
* \note This work was funded by the Netherlands Organisation for
* Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
*
*/
template< class TScalarType = float, unsigned int NDimensions = 3,
          class TParentImageFilter = TranslationTransform< TScalarType, NDimensions > >
class GPUTranslationTransform :
  public TParentImageFilter,
  public GPUTranslationTransformBase< TScalarType, NDimensions >
{
public:
  /** Standard class typedefs. */
  typedef GPUTranslationTransform Self;
  typedef TParentImageFilter      CPUSuperclass;
  typedef GPUMatrixOffsetTransformBase< TScalarType,
                                        NDimensions,
                                        NDimensions > GPUSuperclass;

  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUTranslationTransform, CPUSuperclass );

  /** Typedefs */
  typedef typename GPUSuperclass::CPUOutputVectorType CPUOutputVectorType;

  /**  */
  virtual const CPUOutputVectorType & GetCPUOffset( void ) const { return this->GetOffset(); }

protected:
  GPUTranslationTransform() {}
  virtual ~GPUTranslationTransform() {}

private:
  GPUTranslationTransform( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );        // purposely not implemented
};

/** \class GPUTranslationTransformFactory
* \brief Object Factory implementation for GPUTranslationTransform
*/
class GPUTranslationTransformFactory : public ObjectFactoryBase
{
public:
  typedef GPUTranslationTransformFactory Self;
  typedef ObjectFactoryBase              Superclass;
  typedef SmartPointer< Self >           Pointer;
  typedef SmartPointer< const Self >     ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char * GetDescription() const { return "A Factory for GPUTranslationTransform"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUTranslationTransformFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPUTranslationTransformFactory::Pointer factory =
      GPUTranslationTransformFactory::New();
    ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPUTranslationTransformFactory( const Self & ); // purposely not implemented
  void operator=( const Self & );                 // purposely not implemented

#define OverrideTranslationTransformTypeMacro( st, dm )                   \
  {                                                                       \
    this->RegisterOverride(                                               \
      typeid( TranslationTransform< st, dm > ).name(),                    \
      typeid( GPUTranslationTransform< st, dm > ).name(),                 \
      "GPU TranslationTransform Override",                                \
      true,                                                               \
      CreateObjectFunction< GPUTranslationTransform< st, dm > >::New() ); \
  }

  GPUTranslationTransformFactory()
  {
    if ( IsGPUAvailable() )
    {
      OverrideTranslationTransformTypeMacro( float, 1 );
      //OverrideTranslationTransformTypeMacro(double, 1);

      OverrideTranslationTransformTypeMacro( float, 2 );
      //OverrideTranslationTransformTypeMacro(double, 2);

      OverrideTranslationTransformTypeMacro( float, 3 );
      //OverrideTranslationTransformTypeMacro(double, 3);
    }
  }
};
} // end namespace itk

#endif /* __itkGPUTranslationTransform_h */
