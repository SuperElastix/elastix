/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUAdvancedTranslationTransform_h
#define __itkGPUAdvancedTranslationTransform_h

#include "itkAdvancedTranslationTransform.h"
#include "itkGPUTranslationTransformBase.h"

namespace itk
{
/** \class GPUAdvancedTranslationTransform
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
          class TParentImageFilter = AdvancedTranslationTransform< TScalarType, NDimensions > >
class GPUAdvancedTranslationTransform :
  public TParentImageFilter,
  public GPUTranslationTransformBase< TScalarType, NDimensions >
{
public:
  /** Standard class typedefs. */
  typedef GPUAdvancedTranslationTransform Self;
  typedef TParentImageFilter              CPUSuperclass;
  typedef GPUMatrixOffsetTransformBase< TScalarType,
                                        NDimensions,
                                        NDimensions > GPUSuperclass;

  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUAdvancedTranslationTransform, CPUSuperclass );

  /** Typedefs */
  typedef typename GPUSuperclass::CPUOutputVectorType CPUOutputVectorType;

  /**  */
  virtual const CPUOutputVectorType & GetCPUOffset( void ) const { return this->GetOffset(); }

protected:
  GPUAdvancedTranslationTransform() {}
  virtual ~GPUAdvancedTranslationTransform() {}

private:
  GPUAdvancedTranslationTransform( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );                // purposely not implemented
};

/** \class GPUAdvancedTranslationTransformFactory
* \brief Object Factory implementation for GPUAdvancedTranslationTransform
*/
class GPUAdvancedTranslationTransformFactory : public ObjectFactoryBase
{
public:
  typedef GPUAdvancedTranslationTransformFactory Self;
  typedef ObjectFactoryBase                      Superclass;
  typedef SmartPointer< Self >                   Pointer;
  typedef SmartPointer< const Self >             ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char * GetDescription() const { return "A Factory for GPUAdvancedTranslationTransform"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUAdvancedTranslationTransformFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPUAdvancedTranslationTransformFactory::Pointer factory =
      GPUAdvancedTranslationTransformFactory::New();
    ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPUAdvancedTranslationTransformFactory( const Self & ); // purposely not implemented
  void operator=( const Self & );                         // purposely not implemented

#define OverrideAdvancedTranslationTransformTypeMacro( st, dm )                   \
  {                                                                               \
    this->RegisterOverride(                                                       \
      typeid( AdvancedTranslationTransform< st, dm > ).name(),                    \
      typeid( GPUAdvancedTranslationTransform< st, dm > ).name(),                 \
      "GPU AdvancedTranslationTransform Override",                                \
      true,                                                                       \
      CreateObjectFunction< GPUAdvancedTranslationTransform< st, dm > >::New() ); \
  }

  GPUAdvancedTranslationTransformFactory()
  {
    if ( IsGPUAvailable() )
    {
      OverrideAdvancedTranslationTransformTypeMacro( float, 1 );
      //OverrideAdvancedTranslationTransformTypeMacro(double, 1);

      OverrideAdvancedTranslationTransformTypeMacro( float, 2 );
      //OverrideAdvancedTranslationTransformTypeMacro(double, 2);

      OverrideAdvancedTranslationTransformTypeMacro( float, 3 );
      //OverrideAdvancedTranslationTransformTypeMacro(double, 3);
    }
  }
};
} // end namespace itk

#endif /* __itkGPUAdvancedTranslationTransform_h */
