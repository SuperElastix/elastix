/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUEuler3DTransform_h
#define __itkGPUEuler3DTransform_h

#include "itkEuler3DTransform.h"
#include "itkGPUMatrixOffsetTransformBase.h"

namespace itk
{
/** \class GPUEuler3DTransform
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
          class TParentImageFilter = Euler3DTransform< TScalarType > >
class GPUEuler3DTransform :
  public TParentImageFilter,
  public GPUMatrixOffsetTransformBase< TScalarType, 3, 3 >
{
public:
  /** Standard class typedefs. */
  typedef GPUEuler3DTransform                               Self;
  typedef TParentImageFilter                                CPUSuperclass;
  typedef GPUMatrixOffsetTransformBase< TScalarType, 3, 3 > GPUSuperclass;
  typedef SmartPointer< Self >                              Pointer;
  typedef SmartPointer< const Self >                        ConstPointer;

  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUEuler3DTransform, CPUSuperclass );

  /** Typedefs */
  typedef typename GPUSuperclass::CPUMatrixType        CPUMatrixType;
  typedef typename GPUSuperclass::CPUInverseMatrixType CPUInverseMatrixType;
  typedef typename GPUSuperclass::CPUOutputVectorType  CPUOutputVectorType;

  /**  */
  virtual const CPUMatrixType &        GetCPUMatrix( void ) const { return this->GetMatrix(); }
  virtual const CPUInverseMatrixType & GetCPUInverseMatrix( void ) const { return this->GetInverseMatrix(); }
  virtual const CPUOutputVectorType &  GetCPUOffset( void ) const { return this->GetOffset(); }

protected:
  GPUEuler3DTransform() {}
  virtual ~GPUEuler3DTransform() {}

private:
  GPUEuler3DTransform( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );    // purposely not implemented
};

/** \class GPUEuler3DTransformFactory
* \brief Object Factory implementation for GPUEuler3DTransform
*/
class GPUEuler3DTransformFactory : public ObjectFactoryBase
{
public:
  typedef GPUEuler3DTransformFactory Self;
  typedef ObjectFactoryBase          Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char * GetDescription() const { return "A Factory for GPUEuler3DTransform"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUEuler3DTransformFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPUEuler3DTransformFactory::Pointer factory =
      GPUEuler3DTransformFactory::New();
    ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPUEuler3DTransformFactory( const Self & ); // purposely not implemented
  void operator=( const Self & );             // purposely not implemented

#define OverrideEuler3DTransformTypeMacro( st )                   \
  {                                                               \
    this->RegisterOverride(                                       \
      typeid( Euler3DTransform< st > ).name(),                    \
      typeid( GPUEuler3DTransform< st > ).name(),                 \
      "GPU Euler3DTransform Override",                            \
      true,                                                       \
      CreateObjectFunction< GPUEuler3DTransform< st > >::New() ); \
  }

  GPUEuler3DTransformFactory()
  {
    if ( IsGPUAvailable() )
    {
      OverrideEuler3DTransformTypeMacro( float );
      //OverrideEuler3DTransformTypeMacro(double);

      OverrideEuler3DTransformTypeMacro( float );
      //OverrideEuler3DTransformTypeMacro(double);

      OverrideEuler3DTransformTypeMacro( float );
      //OverrideEuler3DTransformTypeMacro(double);
    }
  }
};
} // end namespace itk

#endif /* __itkGPUEuler3DTransform_h */
