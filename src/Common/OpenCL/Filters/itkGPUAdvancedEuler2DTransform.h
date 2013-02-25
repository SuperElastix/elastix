/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUAdvancedEuler2DTransform_h
#define __itkGPUAdvancedEuler2DTransform_h

#include "itkAdvancedEuler2DTransform.h"
#include "itkGPUMatrixOffsetTransformBase.h"

namespace itk
{
/** \class GPUAdvancedEuler2DTransform
* \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
* Department of Radiology, Leiden, The Netherlands
*
* This implementation was taken from elastix (http://elastix.isi.uu.nl/).
*
* \note This work was funded by the Netherlands Organisation for
* Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
*
*/
template< class TScalarType = float,
          class TParentImageFilter = AdvancedEuler2DTransform< TScalarType > >
class GPUAdvancedEuler2DTransform :
  public TParentImageFilter,
  public GPUMatrixOffsetTransformBase< TScalarType, 2, 2 >
{
public:
  /** Standard class typedefs. */
  typedef GPUAdvancedEuler2DTransform                       Self;
  typedef TParentImageFilter                                CPUSuperclass;
  typedef GPUMatrixOffsetTransformBase< TScalarType, 2, 2 > GPUSuperclass;
  typedef SmartPointer< Self >                              Pointer;
  typedef SmartPointer< const Self >                        ConstPointer;

  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUAdvancedEuler2DTransform, CPUSuperclass );

  /** Typedefs */
  typedef typename GPUSuperclass::CPUMatrixType        CPUMatrixType;
  typedef typename GPUSuperclass::CPUInverseMatrixType CPUInverseMatrixType;
  typedef typename GPUSuperclass::CPUOutputVectorType  CPUOutputVectorType;

  /**  */
  virtual const CPUMatrixType &        GetCPUMatrix( void ) const { return this->GetMatrix(); }
  virtual const CPUInverseMatrixType & GetCPUInverseMatrix( void ) const { return this->GetInverseMatrix(); }
  virtual const CPUOutputVectorType &  GetCPUOffset( void ) const { return this->GetOffset(); }

protected:
  GPUAdvancedEuler2DTransform() {}
  virtual ~GPUAdvancedEuler2DTransform() {}

private:
  GPUAdvancedEuler2DTransform( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );            // purposely not implemented
};

/** \class GPUAdvancedEuler2DTransformFactory
* \brief Object Factory implementation for GPUAdvancedEuler2DTransform
*/
class GPUAdvancedEuler2DTransformFactory : public ObjectFactoryBase
{
public:
  typedef GPUAdvancedEuler2DTransformFactory Self;
  typedef ObjectFactoryBase                  Superclass;
  typedef SmartPointer< Self >               Pointer;
  typedef SmartPointer< const Self >         ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char * GetDescription() const { return "A Factory for GPUAdvancedEuler2DTransform"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUAdvancedEuler2DTransformFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPUAdvancedEuler2DTransformFactory::Pointer factory =
      GPUAdvancedEuler2DTransformFactory::New();
    ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPUAdvancedEuler2DTransformFactory( const Self & ); // purposely not implemented
  void operator=( const Self & );                     // purposely not implemented

#define OverrideAdvancedEuler2DTransformTypeMacro( st )                   \
  {                                                                       \
    this->RegisterOverride(                                               \
      typeid( AdvancedEuler2DTransform< st > ).name(),                    \
      typeid( GPUAdvancedEuler2DTransform< st > ).name(),                 \
      "GPU AdvancedEuler2DTransform Override",                            \
      true,                                                               \
      CreateObjectFunction< GPUAdvancedEuler2DTransform< st > >::New() ); \
  }

  GPUAdvancedEuler2DTransformFactory()
  {
    if ( IsGPUAvailable() )
    {
      OverrideAdvancedEuler2DTransformTypeMacro( float );
      //OverrideAdvancedEuler2DTransformTypeMacro( double );

      OverrideAdvancedEuler2DTransformTypeMacro( float );
      //OverrideAdvancedEuler2DTransformTypeMacro( double );

      OverrideAdvancedEuler2DTransformTypeMacro( float );
      //OverrideAdvancedEuler2DTransformTypeMacro( double );
    }
  }
};
} // end namespace itk

#endif /* __itkGPUAdvancedEuler2DTransform_h */
