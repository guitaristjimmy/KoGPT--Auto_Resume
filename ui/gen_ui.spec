# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['gen_ui.py'],
             pathex=['C:\\Users\\K\\Desktop\\I_SW\\Python_Note\\gpt-2\\ui'],
             binaries=[],
             datas=[('C:\\ProgramData\\Anaconda3\\envs\\gpt-2\\Lib\\site-packages\\mxnet', 'mxnet'),
                    ('C:\ProgramData\Anaconda3\envs\gpt-2\Lib\site-packages\sacremoses', 'sacremoses'),
                    ('./gen_m.tar', '.'),
                    ('./vocab.spiece', '.')],
             hiddenimports=["ctypes"],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='gen_ui',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='gen_ui')
